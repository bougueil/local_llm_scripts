# local_llm_scripts
run local llm with elixir, tested with `cuda13`

* extract subtitles from a movie and store them in a file

```bash
XLA_TARGET=cuda13 ffmsrt.exs my_movie.mp4
```
