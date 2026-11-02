Fake tools for tests of a controller and its Sparks. No network, no GPU.
Put this folder first on PATH and set FAKE_HOSTS to a folder that holds one
folder per host: $FAKE_HOSTS/host-a is the home folder of host-a.

A file named known in a home folder means that the controller has the host key
of that host. With no such file, ssh and rsync fail as the real tools do.
A file named driver holds the NVIDIA driver version of that host.
A file named corrupt makes rsync change the files it copies to that host.
Each ssh and rsync call adds one row to $FAKE_HOSTS/calls.log.
