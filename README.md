# A large-D closure for asymmetric kinetic vector-spin systems

<p align="center">
  <a href="https://mcbal.github.io/post/a-large-d-closure-for-asymmetric-kinetic-vector-spin-systems/"><img src="assets/featured.jpg" alt="" width="50%"></a>
</p>

## Contents

- [`blog/`](blog/): the post's numerical checks in plain NumPy, together with the JSON results behind each figure. See [`blog/README.md`](blog/README.md) for the scripts and how to run them.

## Quick start

```bash
cd blog
uv run --with numpy,matplotlib python convergence.py --plot   # redraw Figs. 2–3 from the included results
uv run --with numpy,matplotlib python convergence.py          # rerun the sweep (about 6 min on 4 cores)
```

## Related

- [Spin-Model Transformers](https://mcbal.github.io/post/spin-model-transformers/)
- [Nonequilibrium Dynamics in Spin-Model Transformers](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/) and [mcbal/neqnn](https://github.com/mcbal/neqnn)

## License

MIT, see [LICENSE](LICENSE).
