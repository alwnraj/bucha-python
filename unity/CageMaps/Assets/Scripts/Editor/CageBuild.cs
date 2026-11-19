// Editor menu and batch-mode entry: build a map into Assets/Scenes/Cage.unity and check it.
// Batch: Unity -batchmode -nographics -projectPath unity/CageMaps -executeMethod CageBuild.BuildAndCheck -map seed-1.json
using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class CageBuild
{
    const string ScenePath = "Assets/Scenes/Cage.unity";

    [MenuItem("Cage/Build scene from map...")]
    static void BuildFromMenu()
    {
        var path = EditorUtility.OpenFilePanel("Map JSON", Path.Combine(Application.streamingAssetsPath, "maps"), "json");
        if (!string.IsNullOrEmpty(path)) Debug.Log(Check(BuildScene(Path.GetFileName(path))));
    }

    public static void BuildAndCheck()
    {
        var args = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(args, "-map");
        var file = i >= 0 ? args[i + 1] : "scatter-1.json";
        int failures;
        try
        {
            var report = Check(BuildScene(file));
            Debug.Log(report);
            failures = report.Split('\n').Count(l => l.StartsWith("FAIL"));
        }
        catch (Exception e) { Debug.LogException(e); failures = 1; }
        EditorApplication.Exit(failures == 0 ? 0 : 1);
    }

    // Batch: ... -executeMethod CageBuild.PlayCheck  (needs the scene built first)
    // Enters Play mode, drives the tracked body into the first object and back out, and checks
    // that contact start and end are reported. Also reports the VRPN connection status.
    public static void PlayCheck()
    {
        EditorSceneManager.OpenScene(ScenePath);
        EditorSettings.enterPlayModeOptionsEnabled = true;
        EditorSettings.enterPlayModeOptions = EnterPlayModeOptions.DisableDomainReload;
        int step = 0, failures = 0;
        double t0 = 0;
        TrackedBody tb = null; MapItem target = null;
        EditorApplication.update += () =>
        {
            if (!EditorApplication.isPlaying) return;
            double now = EditorApplication.timeSinceStartup;
            if (step == 0) { t0 = now; step = 1; }
            if (step == 1 && now - t0 > 1)
            {
                step = 2; t0 = now;
                tb = UnityEngine.Object.FindAnyObjectByType<TrackedBody>();
                tb.vrpn = null;  // drive it from this test, not from live mocap
                target = UnityEngine.Object.FindObjectsByType<MapItem>(FindObjectsSortMode.None).OrderBy(m => m.id).First();
                var b = new Bounds(target.transform.position, Vector3.zero);
                foreach (var r in target.GetComponentsInChildren<Renderer>()) b.Encapsulate(r.bounds);
                tb.GetComponent<Rigidbody>().position = new Vector3(b.center.x, 0.1f, b.center.z);
            }
            if (step == 2 && now - t0 > 1)
            {
                step = 3; t0 = now;
                bool ok = tb.touching.Contains(target);
                Debug.Log($"{(ok ? "ok  " : "FAIL")} contact start with {target.id} reported");
                if (!ok) failures++;
                tb.GetComponent<Rigidbody>().position = new Vector3(0.2f, 0.3f, 0.2f);
            }
            var client = UnityEngine.Object.FindAnyObjectByType<VrpnClient>();
            if (step == 3 && now - t0 > 1 && (client.Status != "not started" || now - t0 > 8))
            {
                bool ok = tb.touching.Count == 0;
                Debug.Log($"{(ok ? "ok  " : "FAIL")} contact end reported");
                if (!ok) failures++;
                var vrpn = UnityEngine.Object.FindAnyObjectByType<VrpnClient>();
                string pose = vrpn.TryGet(tb.bodyName, out var p)
                    ? $"{tb.bodyName} motive pos ({p.position[0]:F3}, {p.position[1]:F3}, {p.position[2]:F3}) quat ({p.quaternion[0]:F3}, {p.quaternion[1]:F3}, {p.quaternion[2]:F3}, {p.quaternion[3]:F3})"
                      + $" -> cage pos {tb.MotiveToCage(p.position, p.quaternion, out var cr):F3} yaw {cr.eulerAngles.y:F1}"
                    : "no pose received";
                Debug.Log($"VRPN: {vrpn.Status}; {pose}");
                EditorApplication.Exit(failures);
            }
        };
        EditorApplication.EnterPlaymode();
    }

    static GameObject BuildScene(string mapFile)
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        var map = new GameObject("Map");
        var loader = map.AddComponent<MapLoader>();
        loader.mapFile = mapFile;
        var data = MapLoader.Build(Path.Combine(Application.streamingAssetsPath, "maps", mapFile), map.transform);

        var cam = Camera.main.transform;  // look down the cage from behind one short side
        cam.position = new Vector3(data.floor.size_x / 2, 7f, -3f);
        cam.LookAt(new Vector3(data.floor.size_x / 2, 0, data.floor.size_z * 0.45f));

        var vrpn = new GameObject("Mocap (VRPN)").AddComponent<VrpnClient>();
        var tracked = GameObject.CreatePrimitive(PrimitiveType.Cube);
        tracked.name = "Tracked modalai";
        UnityEngine.Object.DestroyImmediate(tracked.GetComponent<BoxCollider>());
        tracked.AddComponent<Rigidbody>();
        tracked.AddComponent<BoxCollider>();
        var tb = tracked.AddComponent<TrackedBody>();
        tb.vrpn = vrpn;
        tracked.transform.localScale = Vector3.one;
        tracked.transform.position = new Vector3(data.floor.size_x / 2, 0.3f, 0.3f);
        var shape = GameObject.CreatePrimitive(PrimitiveType.Cube);  // visible box matching the collider
        UnityEngine.Object.DestroyImmediate(shape.GetComponent<BoxCollider>());
        shape.transform.SetParent(tracked.transform, false);
        shape.transform.localScale = tb.size;
        UnityEngine.Object.DestroyImmediate(tracked.GetComponent<MeshRenderer>());
        UnityEngine.Object.DestroyImmediate(tracked.GetComponent<MeshFilter>());

        Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, ScenePath);
        return map;
    }

    // Each object must be found by a box query at its own position, and the floor must stop a downward ray.
    static string Check(GameObject map)
    {
        Physics.SyncTransforms();
        var lines = new System.Collections.Generic.List<string>();
        foreach (var item in map.GetComponentsInChildren<MapItem>())
        {
            var b = new Bounds(item.transform.position, Vector3.zero);
            foreach (var r in item.GetComponentsInChildren<Renderer>()) b.Encapsulate(r.bounds);
            var hits = Physics.OverlapBox(b.center, b.extents * 0.9f);
            bool found = hits.Any(h => h.GetComponentInParent<MapItem>() == item);
            lines.Add($"{(found ? "ok  " : "FAIL")} {item.id} {item.asset} centre {b.center:F2} size {b.size:F2} base y {b.min.y:F3}");
            if (Mathf.Abs(b.min.y) > 0.01f) lines.Add($"FAIL {item.id} base not on floor");
            var f = item.footprint;
            if (f == null || f.Length != 4 || Mathf.Abs(b.min.x - f[0]) > 0.01f || Mathf.Abs(b.min.z - f[1]) > 0.01f
                || Mathf.Abs(b.max.x - f[2]) > 0.01f || Mathf.Abs(b.max.z - f[3]) > 0.01f)
                lines.Add($"FAIL {item.id} footprint {b.min.x:F2},{b.min.z:F2}..{b.max.x:F2},{b.max.z:F2} differs from map file");
        }
        foreach (var zone in new[] { "Start", "Goal" })
        {
            var z = map.transform.Find(zone);
            lines.Add($"{(z != null && z.GetComponent<Collider>() == null ? "ok  " : "FAIL")} {zone} zone shown, no collider");
        }
        bool floor = Physics.Raycast(new Vector3(0.2f, 1, 0.2f), Vector3.down, out var hit, 5) && hit.collider.name == "Floor";
        lines.Add($"{(floor ? "ok  " : "FAIL")} floor ray hit at y {hit.point.y:F3}");
        if (lines.Count < 2) lines.Add("FAIL no objects");
        return "Cage map check:\n" + string.Join("\n", lines);
    }
}
