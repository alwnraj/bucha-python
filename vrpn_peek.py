"""Minimal read-only VRPN client: print a tracker's Pos_Quat reports for a few seconds.

Read-only: it never changes anything in Motive. No packages needed.
Usage: python3 vrpn_peek.py modalai 192.168.1.42 [seconds]
Connects over TCP, exchanges the VRPN cookie, describes the tracker name and the
Pos_Quat message type (VRPN servers only send types the client has described),
then decodes position (metres) and quaternion (x, y, z, w).
"""
import socket
import struct
import sys
import time

COOKIE = b"vrpn: ver. 07.35  0".ljust(24, b"\0")
SENDER_DESCRIPTION, TYPE_DESCRIPTION = -1, -2


def pad8(n):
    return (n + 7) // 8 * 8


def message(sender, mtype, payload):
    now = time.time()
    header = struct.pack(">iiiii", 24 + len(payload), int(now), int(now % 1 * 1e6), sender, mtype) + b"\0" * 4
    return header + payload + b"\0" * (pad8(len(payload)) - len(payload))


def description(local_id, kind, name):
    text = name.encode() + b"\0"
    return message(local_id, kind, struct.pack(">i", len(text)) + text)


def main():
    body, host = sys.argv[1], sys.argv[2]
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 5
    s = socket.create_connection((host, 3883), timeout=5)
    s.sendall(COOKIE)
    cookie = b""
    while len(cookie) < 24:
        cookie += s.recv(24 - len(cookie))
    print("server cookie:", cookie.rstrip(b"\0").decode(errors="replace"))
    s.sendall(description(0, SENDER_DESCRIPTION, body) + description(0, TYPE_DESCRIPTION, "vrpn_Tracker Pos_Quat"))
    senders, types, buffer, reports, first, last = {}, {}, b"", 0, None, None
    end = time.time() + seconds
    s.settimeout(0.5)
    while time.time() < end:
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            print("server closed the connection")
            break
        buffer += chunk
        while len(buffer) >= 24:
            length, sec, usec, sender, mtype = struct.unpack(">iiiii", buffer[:20])
            total = 24 + pad8(length - 24)
            if len(buffer) < total:
                break
            payload, buffer = buffer[24:length], buffer[total:]
            if mtype in (SENDER_DESCRIPTION, TYPE_DESCRIPTION):
                (size,) = struct.unpack(">i", payload[:4])
                name = payload[4:4 + size].rstrip(b"\0").decode(errors="replace")
                (senders if mtype == SENDER_DESCRIPTION else types)[sender] = name
            elif types.get(mtype) == "vrpn_Tracker Pos_Quat" and senders.get(sender) == body:
                sensor = struct.unpack(">i", payload[:4])[0]
                position = struct.unpack(">3d", payload[8:32])
                quaternion = struct.unpack(">4d", payload[32:64])
                reports += 1
                last = (sensor, position, quaternion)
                first = first or last
    s.close()
    print("senders seen:", sorted(set(senders.values())))
    print(f"{body} Pos_Quat reports in {seconds:.0f} s: {reports} ({reports / seconds:.0f} per second)")
    if last:
        for label, (sensor, p, q) in (("first", first), ("last", last)):
            print(f"  {label}: position x={p[0]:.4f} y={p[1]:.4f} z={p[2]:.4f} m | "
                  f"quaternion x={q[0]:.4f} y={q[1]:.4f} z={q[2]:.4f} w={q[3]:.4f}")


if __name__ == "__main__":
    main()
