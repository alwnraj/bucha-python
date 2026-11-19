// A real tracked thing in the cage, shown as a translucent box that follows its mocap pose.
// It is a kinematic trigger, so it never pushes map objects, but every touch with one is reported.
// With no mocap data (away from the lab) it can be driven with the arrow keys / PageUp / PageDown.
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody), typeof(BoxCollider))]
public class TrackedBody : MonoBehaviour
{
    public VrpnClient vrpn;
    public string bodyName = "modalai";
    public Vector3 size = new Vector3(0.35f, 0.15f, 0.35f);

    // Measured 2026-11-18 with modalai at the front, right and back edge middles (about +-0.15 m):
    // Motive +Z points to the cage front, Motive -X to the right side, origin near the centre.
    // Cage X 0 is the left side (inferred from the 22 ft width; not measured), Z 0 the back end.
    [Header("Motive -> cage frame (measured 2026-11-18, about +-0.15 m)")]
    [Tooltip("Where Motive's origin sits in the cage frame (m).")]
    public Vector3 motiveOriginInCage = new Vector3(3.01f, 0, 4.74f);
    [Tooltip("Rotation of Motive's axes about Y relative to the cage frame (deg).")]
    public float motiveYawInCage = 0;
    [Tooltip("Motive is right-handed and Unity left-handed: X is mirrored when on.")]
    public bool mirrorX = true;

    [Header("Keyboard driving when no mocap data")]
    public float keySpeed = 1.0f;

    public readonly HashSet<MapItem> touching = new HashSet<MapItem>();
    public float lastPoseAge = float.PositiveInfinity;
    Rigidbody body;

    void Awake()
    {
        body = GetComponent<Rigidbody>();
        body.isKinematic = true;
        body.useGravity = false;
        body.interpolation = RigidbodyInterpolation.Interpolate;
        var box = GetComponent<BoxCollider>();
        box.isTrigger = true;
        box.size = size;
    }

    // Motive (right-handed, Y up) -> Unity (left-handed, Y up), then into the cage frame.
    public Vector3 MotiveToCage(double[] p, double[] q, out Quaternion rot)
    {
        MotiveToUnity(p, q, mirrorX, out var pos, out var r);
        var yaw = Quaternion.Euler(0, motiveYawInCage, 0);
        rot = yaw * r;
        return motiveOriginInCage + yaw * pos;
    }

    public static void MotiveToUnity(double[] p, double[] q, bool mirrorX, out Vector3 pos, out Quaternion rot)
    {
        float s = mirrorX ? -1 : 1;
        pos = new Vector3(s * (float)p[0], (float)p[1], (float)p[2]);
        // A rotation seen in a mirror (X flipped) keeps its angle; its axis becomes (ax, -ay, -az).
        rot = mirrorX ? new Quaternion((float)q[0], -(float)q[1], -(float)q[2], (float)q[3])
                      : new Quaternion((float)q[0], (float)q[1], (float)q[2], (float)q[3]);
    }

    void FixedUpdate()
    {
        if (vrpn != null && vrpn.TryGet(bodyName, out var pose))
        {
            lastPoseAge = (float)(System.DateTime.UtcNow - pose.received).TotalSeconds;
            body.MovePosition(MotiveToCage(pose.position, pose.quaternion, out var rot));
            body.MoveRotation(rot);
            return;
        }
        var move = new Vector3(Input.GetAxisRaw("Horizontal"), 0, Input.GetAxisRaw("Vertical"));
        if (Input.GetKey(KeyCode.PageUp)) move.y += 1;
        if (Input.GetKey(KeyCode.PageDown)) move.y -= 1;
        body.MovePosition(body.position + move * keySpeed * Time.fixedDeltaTime);
    }

    void OnTriggerEnter(Collider other)
    {
        var item = other.GetComponentInParent<MapItem>();
        if (item != null && touching.Add(item)) Debug.Log($"CONTACT start {bodyName} -> {item.id} ({item.asset}) at {transform.position}");
    }

    void OnTriggerExit(Collider other)
    {
        var item = other.GetComponentInParent<MapItem>();
        if (item == null) return;
        // An object has several colliders; only report the end when none of them still overlap.
        foreach (var c in Physics.OverlapBox(transform.position, size / 2, transform.rotation))
            if (c != other && c.GetComponentInParent<MapItem>() == item) return;
        if (touching.Remove(item)) Debug.Log($"CONTACT end {bodyName} -> {item.id} ({item.asset})");
    }

    void OnGUI()
    {
        var src = vrpn == null ? "keyboard" : vrpn.Status + (lastPoseAge < 1 ? "" : " (no recent pose: keyboard)");
        var names = new List<string>();
        foreach (var t in touching) names.Add(t.id);
        GUILayout.Label($"{bodyName}: {src}  pos {transform.position:F2}  touching: {(names.Count == 0 ? "none" : string.Join(", ", names))}");
    }
}
