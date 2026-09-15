# MDM Runtime Resources

This directory makes MDM inference self-contained inside `bvh_processing`.
Large model files are tracked with Git LFS.

Required layout:

```text
checkpoint/args.json
checkpoint/model000600000.pt
humanml/Mean.npy
humanml/Std.npy
humanml/new_joints/000021.npy
text_encoder/config.json
text_encoder/model.safetensors
text_encoder/tokenizer.json
text_encoder/tokenizer_config.json
text_encoder/vocab.txt
```

Model SHA-256 values:

- `model000600000.pt`: `195664bed72143e071acef4c97ac1aa67f8aa00f57fdc691248e98d715356d92`
- `model.safetensors`: `5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`
