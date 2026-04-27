#! /usr/bin/env elixir
Mix.install([
  {:xla, "~> 0.10.0", override: true},
  {:nx, "~> 0.11.0", override: true},
  {:bumblebee, git: "https://github.com/elixir-nx/bumblebee.git"},
  {:exla, "~> 0.11.0", override: true}
])

Application.put_env(:exla, :clients,
  host: [platform: :host],
  cuda: [
    platform: :cuda,
    default_device_id: 0,
    memory_fraction: 0.9
  ]
)

defmodule Main do
  @moduledoc false

  # @model_speech2text "openai/whisper-tiny"
  @model_speech2text "openai/whisper-large-v3-turbo"
  @debug_chunk true

  defp doc,
    do: """
    Synopsis:
    Convert a movie file into a srt file (subtitle) using the local speech2text: #{@model_speech2text}.
    Usage:
    $ ffmsrt.exs # help
    $ ffmsrt.exs /tmp/english_spoken_video.mp4
    """

  def main([audio_path | _args]) do
    if !File.exists?(audio_path) do
      if String.contains?(audio_path, " ") do
        IO.puts(
          ">> the path contains probably space, enclose the path with \"\"\npath: #{audio_path}."
        )
      else
        IO.puts(">> no audio_path found for #{audio_path}.")
      end

      System.halt()
    end

    Nx.global_default_backend({EXLA.Backend, client: :cuda})
    model_name = @model_speech2text
    {:ok, model} = Bumblebee.load_model({:hf, model_name}, backend: EXLA.Backend)
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_name})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, model_name})

    serving =
      Bumblebee.Audio.speech_to_text_whisper(
        model,
        featurizer,
        tokenizer,
        generation_config,
        chunk_num_seconds: 30,
        timestamps: :segments,
        defn_options: [compiler: EXLA]
      )

    %{chunks: chunks} = Nx.Serving.run(serving, {:file, audio_path})

    Stream.transform(chunks, 1, fn chunk, id -> {[build_srt(chunk, id)], id + 1} end)
    |> Stream.into(File.stream!(audio_path <> ".srt"))
    |> Stream.run()

    IO.puts("\nsrt file #{audio_path <> ".srt"} created.")
  end

  def main(_), do: IO.puts(doc())

  defp build_srt(%{text: text, start_timestamp_seconds: start, end_timestamp_seconds: fend}, id)
       when is_nil(start) or is_nil(fend) or is_nil(text) do
    perrors("** chunk error #2 (#{id}), #{start} #{fend} #{text}")
    ""
  end

  defp build_srt(
         %{text: text, start_timestamp_seconds: start, end_timestamp_seconds: fend} = chk,
         id
       ) do
    sec_dur = fend - start

    cond do
      sec_dur < 0 or sec_dur > 20 ->
        invalid_chunk_time(sec_dur, chk, id)
        ""

      sec_dur > 1 ->
        srt_element(id, start, fend, text)

      true ->
        srt_element(id, start, heuristic_subtitle_end(start, text), text)
    end
  end

  defp build_srt(chunk, id) do
    perrors("** chunk error #1 (#{id}), #{inspect(chunk)}")
    ""
  end

  defp heuristic_subtitle_end(start, txt) do
    start + String.length(txt) / 20
  end

  defp invalid_chunk_time(dur, chk, id) when dur > 0 do
    %{text: text, start_timestamp_seconds: start, end_timestamp_seconds: fend} = chk

    perrors("** chunk error #3.2 (#{id}), #{hms_ms(start)} <-> #{hms_ms(fend)} #{text}")

    # FIX TBD
    ""
  end

  defp invalid_chunk_time(
         _dur,
         %{text: text, start_timestamp_seconds: start, end_timestamp_seconds: fend},
         id
       ) do
    perrors("** chunk error #3.1 (#{id}), #{hms_ms(start)} <-> #{hms_ms(fend)} #{text}")
    ""
  end

  defp srt_element(id, start, fend, text) do
    """
    #{id}
    #{hms_ms(start)} --> #{hms_ms(fend)}
    #{text}

    """
  end

  defp perrors(err_string) do
    @debug_chunk && IO.puts(err_string)
  end

  defp hms_ms(seconds_float) do
    seconds = trunc(seconds_float)
    ms = round((seconds_float - seconds) * 1000_000)

    Time.add(Time.new!(0, 0, 0, {ms, 3}), seconds, :second)
    |> Time.to_string()
    |> String.replace(".", ",")
  end
end

Main.main(System.argv())
