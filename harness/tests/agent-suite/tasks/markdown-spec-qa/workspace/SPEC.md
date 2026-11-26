# Tidemark sync protocol, version 3

Tidemark keeps a folder on a client in step with a folder on a server. This
document describes version 3 of its wire protocol.

## Connection

A client opens one TCP connection to the server. The default port for
version 3 is 7401. The server sends nothing until the client speaks.

Every message is one frame. A frame starts with a 4-byte big-endian length,
followed by that many bytes of UTF-8 JSON. The largest frame is 65536 bytes,
including the length field. A peer that receives a larger frame closes the
connection with error code E104.

## Handshake

1. The client sends `HELLO` with its protocol version and client name.
2. The server answers with `CHALLENGE`, which holds a 32-byte nonce.
3. The client sends `AUTH` with the nonce signed by its key.
4. The server answers with `READY` or closes the connection with E201.

After `READY`, the client can send `PULL` and `PUSH` messages.

## Cursors

The server gives each change a cursor, a 64-bit counter. A `PULL` names the
last cursor the client has seen. When that cursor is older than the oldest
change the server still keeps, the server answers with error code E117 (stale
cursor). The client must then start a full scan.

The server keeps changes for 14 days.

## Retries

When the connection drops, the client waits and connects again. It waits 2
seconds before the first retry and doubles the wait each time. It tries 6
times. After the last failed try, it reports the folder as offline.

## Error codes

| Code | Meaning |
| --- | --- |
| E104 | Frame too large |
| E117 | Stale cursor |
| E201 | Authentication failed |
| E305 | Folder is locked by another client |

## History

Version 2 used port 7400 and a largest frame of 16384 bytes. It retried 4
times and kept changes for 7 days. Version 3 keeps the same error codes.
