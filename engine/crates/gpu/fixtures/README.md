# CUDA ELF header fixtures

These are the first 64 bytes of real `ptxas` output, copied unchanged. They
exercise header policy only; they are not loadable modules or correctness
fixtures. Adjacent compiler logs recorded `sm_121` and `sm_121a`, respectively.

| File | Full producer image SHA256 | Header SHA256 |
| --- | --- | --- |
| `sm_121.elf-header` | `ac46c5317e3ebd7898b1abb8a1a554e9f7ec5b880d19167b29575f41db042910` | `f4220927f291ce40fb764530b0214b4167155e55558113fa048773593ace9eab` |
| `sm_121a.elf-header` | `2aed71b5e292c9ec19733c77d5644756489f81da39f128744c74c7d33d33173d` | `5155a0f1ff7beda9294211fb17cf51aebe6400ed0eeb360bbeee9958737f0582` |

Both have ELF OSABI `0x41`, ABI version `8`, executable type `2`, machine
`190`, ELF version `1`, and a 64-byte header. Their flags differ:
`sm_121` is `0x06007902`; `sm_121a` is `0x0a007902`. The producer validator
accepts only the first complete encoding, including all target-feature bits.
New CUDA ELF ABI or flag encodings require explicit review and new fixtures.
