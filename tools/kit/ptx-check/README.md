# ptx-check

**Measures:** byte-wise stores to local memory (`st.local.b8`) in each `.entry` and `.func` of `flash.ptx`, candidate against head. A new byte-copy loop in a kernel shows as new stores. It also compares registers per kernel from the matching cubins.

**Run:** `bash tools/kit/ptx-check/ptx-check.sh <candidate dir> <head dir>`. Each folder holds `flash.ptx`, and can hold `flash.cubin` and `BUILD-REVISION`. A staged install folder, `~/.vade/stage/builds/<generation>`, also works.

**Prints:** one line per difference (`MISSING`, `NEW`, `BYTECOPY`, `FEWER`, `REG`), a `PTX DEFINITIONS` summary and a `REGISTERS` line. The last line is `PTX CHECK <rev> clean` (exit 0) or `PTX CHECK <rev> FAIL <entries>` (exit 1). Bad input gives `FAIL input: <reason>` and exit 3. A wrong argument count gives the usage line and exit 2.

**Needs:** bash and Ruby. No GPU. Two 37 MB PTX files take a few seconds.

**Change from the reviewed copy:** none. The file is byte-identical to the reviewed checker ([SOURCES.txt](../SOURCES.txt)). Its header still names its old private path in the usage example.

**Rule exception:** the check runs as inline Ruby, not Rust, until someone ports it.
