These files are synthetic fixtures for tests/host_edges/mod.rs.
The 20-byte ELF headers are not runnable executables.
The flash.ptx.fixture files contain plain text. They are not compiled GPU code.
The tests copy each bundle to its own temporary directory. During the copy,
flash.ptx.fixture becomes engine-cli/flash.ptx, as required by the manifest.
Each test removes its temporary copy when done. The source fixtures stay unchanged.

valid contains a valid minimal input for Bundle::find.
Each other folder changes the field or file named by its folder name.
Host cases cover each target returned by prebuilt::host_target.
Host identity supplies a host hash without its required identity fields.
