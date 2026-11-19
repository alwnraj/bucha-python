// Read-only VRPN tracker client (C# port of vrpn_peek.py). Never sends anything that changes Motive.
// A background thread keeps the latest Pos_Quat per rigid body name; read it with TryGet.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class VrpnClient : MonoBehaviour
{
    public string host = "192.168.1.42";
    public int port = 3883;
    public string[] bodies = { "modalai", "starscream" };

    public struct Pose { public double[] position, quaternion; public DateTime received; }

    readonly Dictionary<string, Pose> latest = new Dictionary<string, Pose>();
    Thread thread;
    volatile bool running;
    public string Status { get; private set; } = "not started";

    public bool TryGet(string body, out Pose pose) { lock (latest) return latest.TryGetValue(body, out pose); }

    void OnEnable() { running = true; thread = new Thread(Run) { IsBackground = true }; thread.Start(); }
    void OnDisable() { running = false; thread?.Join(1000); }

    static int Pad8(int n) => (n + 7) / 8 * 8;

    static byte[] Message(int sender, int type, byte[] payload)
    {
        var msg = new byte[24 + Pad8(payload.Length)];
        double now = (DateTime.UtcNow - DateTime.UnixEpoch).TotalSeconds;
        BinaryPrimitives.WriteInt32BigEndian(msg.AsSpan(0), 24 + payload.Length);
        BinaryPrimitives.WriteInt32BigEndian(msg.AsSpan(4), (int)now);
        BinaryPrimitives.WriteInt32BigEndian(msg.AsSpan(8), (int)(now % 1 * 1e6));
        BinaryPrimitives.WriteInt32BigEndian(msg.AsSpan(12), sender);
        BinaryPrimitives.WriteInt32BigEndian(msg.AsSpan(16), type);
        payload.CopyTo(msg, 24);
        return msg;
    }

    static byte[] Description(int localId, int kind, string name)
    {
        var text = Encoding.ASCII.GetBytes(name + "\0");
        var payload = new byte[4 + text.Length];
        BinaryPrimitives.WriteInt32BigEndian(payload, text.Length);
        text.CopyTo(payload, 4);
        return Message(localId, kind, payload);
    }

    void Run()
    {
        while (running)
        {
            try { Session(); }
            catch (Exception e) { Status = "error: " + e.Message; }
            for (int i = 0; i < 20 && running; i++) Thread.Sleep(100);  // retry after 2 s
        }
    }

    void Session()
    {
        using var client = new TcpClient();
        if (!client.ConnectAsync(host, port).Wait(5000)) throw new TimeoutException($"no answer from {host}:{port}");
        var s = client.GetStream();
        s.ReadTimeout = 2000;
        s.Write(Encoding.ASCII.GetBytes("vrpn: ver. 07.35  0".PadRight(24, '\0')));
        var cookie = new byte[24];
        for (int got = 0; got < 24;) { int n = s.Read(cookie, got, 24 - got); if (n == 0) throw new Exception("closed"); got += n; }
        for (int i = 0; i < bodies.Length; i++) s.Write(Description(i, -1, bodies[i]));
        s.Write(Description(0, -2, "vrpn_Tracker Pos_Quat"));
        Status = "connected: " + Encoding.ASCII.GetString(cookie).TrimEnd('\0');

        var senders = new Dictionary<int, string>();
        var types = new Dictionary<int, string>();
        var buf = new List<byte>();
        var chunk = new byte[65536];
        while (running)
        {
            int n;
            try { n = s.Read(chunk, 0, chunk.Length); }
            catch (System.IO.IOException) { continue; }  // read timeout; keep waiting
            if (n == 0) throw new Exception("server closed the connection");
            buf.AddRange(new ArraySegment<byte>(chunk, 0, n));
            while (buf.Count >= 24)
            {
                var head = buf.GetRange(0, 20).ToArray();
                int length = BinaryPrimitives.ReadInt32BigEndian(head.AsSpan(0));
                int sender = BinaryPrimitives.ReadInt32BigEndian(head.AsSpan(12));
                int type = BinaryPrimitives.ReadInt32BigEndian(head.AsSpan(16));
                int total = 24 + Pad8(length - 24);
                if (buf.Count < total) break;
                var payload = buf.GetRange(24, length - 24).ToArray();
                buf.RemoveRange(0, total);
                if (type == -1 || type == -2)
                {
                    int size = BinaryPrimitives.ReadInt32BigEndian(payload);
                    var name = Encoding.ASCII.GetString(payload, 4, size).TrimEnd('\0');
                    (type == -1 ? senders : types)[sender] = name;
                }
                else if (types.TryGetValue(type, out var tn) && tn == "vrpn_Tracker Pos_Quat"
                         && senders.TryGetValue(sender, out var body) && payload.Length >= 64)
                {
                    var p = new double[3]; var q = new double[4];
                    for (int i = 0; i < 3; i++) p[i] = BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64BigEndian(payload.AsSpan(8 + 8 * i)));
                    for (int i = 0; i < 4; i++) q[i] = BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64BigEndian(payload.AsSpan(32 + 8 * i)));
                    lock (latest) latest[body] = new Pose { position = p, quaternion = q, received = DateTime.UtcNow };
                }
            }
        }
    }
}
