# Resync recipe — Fuchsia vk_radix_sort

When upstream advances and we want to pull a newer revision, follow this
recipe. The vendored tree should always be reproducible from upstream + the
two local patches recorded in `SOURCE.md`.

## Steps

```bash
# 1. Fresh shallow clone of the mirror (or canonical Fuchsia subtree).
cd /tmp
git clone --depth=1 https://github.com/juliusikkala/fuchsia_radix_sort upstream-new

# 2. Diff against the current vendored tree to see drift.
diff -r --brief upstream-new \
    /path/to/harmonyos_3dgs/third_party/fuchsia_vk_radix_sort \
    | grep -v "Only in .*: SOURCE.md\|SYNC.md\|.git"

# 3. Replay our local patches (see SOURCE.md "Local patches"):
#    a) remove `-Os` from the glslangValidator invocation in CMakeLists.txt
#    b) trim the `targets` list in CMakeLists.txt to {arm/bifrost8, nvidia/sm35}
#    c) delete platforms/vk/targets/vendors/{amd, arm/bifrost4, intel}
#    d) trim platforms/vk/target.c to drop extern decls + switch cases for
#       the deleted vendor targets (otherwise linker errors)

# 4. Update vendored copy in-place, update mirror SHA + date in SOURCE.md.

# 5. Smoke-test: rebuild the project and run the link gate test.
cd /path/to/harmonyos_3dgs
cmake --build build --target test_fuchsia_link -j
./build/test_fuchsia_link

# 6. Run the full unit suite to confirm no regression.
ctest --test-dir build --output-on-failure
```

## What signals a successful sync

- `test_fuchsia_link` PASSes (radix_sort_vk_create + destroy round-trip on a
  device created with shaderInt64 + bufferDeviceAddress + vulkanMemoryModel).
- All previously-passing tests in the harmonyos_3dgs suite still pass.
- No new validation-layer warnings during `test_fuchsia_link`.
