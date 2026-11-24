# Third-party components and data

- `ms-swift/` contains a vendored copy of the ModelScope ms-swift project, including its own Apache-2.0 `LICENSE`. The two project-specific GRPO plugins live in `ms-swift/examples/train/grpo/plugin/`.
- `data/` contains the complete local training and evaluation datasets requested for this release. Credential-like query parameters found inside retrieved URLs were replaced with `REDACTED`; the dataset records and splits remain present. Some examples include model-generated tool responses and excerpts from external web pages. Original publishers retain rights to their source content.
- Model weights, checkpoints, API credentials and local runtime caches are not included. Obtain and use external models and APIs under their respective terms.

The Apache-2.0 license at the repository root applies to the repository's project code to the extent the repository owner holds the required rights; third-party components retain their own notices and licenses.
