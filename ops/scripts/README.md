# OS setup and Rust qualification references

Use Rust [`vade up`](../../cli/README.md#multi-spark-source-setup) for product
serving, and `vade down`, `vade status`, `vade logs` and `vade chat` for its owned
lifecycle. Copy the private [CLI configuration example](../../cli/examples/vade.conf)
as described in [operations setup](../README.md).

- [Rust gate guide](../../cli/docs/test.md): `vade test build` and
  `vade test determinism`.
- [Tools reference](tools/README.md): `vade bench` and evaluation commands.
- `spark-setup/`: fresh-DGX-OS preparation helpers; read the
  [system setup guide](../../docs/spark-setup.md) before administrator action.

Keep machine configuration, credentials and generated evidence outside Git.
Historical source manifests describe the original imported artifacts and their
hashes, not the current bytes of files that have since changed.
