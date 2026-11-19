// Builds a cage map (JSON from cage_map.py) into the scene: a floor with a collider and one
// static object per entry, each with mesh colliders. Unity world = the map's "cage" frame.
using System;
using System.IO;
using UnityEngine;

[Serializable] public class MapFloor { public float size_x, size_z; }
[Serializable] public class MapObject { public string id, asset; public float yaw_deg, scale; public float[] matrix, size, footprint; }
[Serializable] public class MapZone { public float[] center; public float size; }
[Serializable] public class CageMapData { public int version, seed; public string frame; public MapFloor floor; public MapZone start, goal; public MapObject[] objects; }

public class MapLoader : MonoBehaviour
{
    [Tooltip("File in Assets/StreamingAssets/maps/")]
    public string mapFile = "scatter-1.json";

    void Start()
    {
        if (transform.childCount == 0) Build(Path.Combine(Application.streamingAssetsPath, "maps", mapFile), transform);
    }

    public static CageMapData Build(string path, Transform root)
    {
        var map = JsonUtility.FromJson<CageMapData>(File.ReadAllText(path));
        for (int i = root.childCount - 1; i >= 0; i--) DestroyImmediate(root.GetChild(i).gameObject);

        // Floor: a 10 cm thick box whose top face is Y = 0, covering the cage.
        var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
        floor.name = "Floor";
        floor.transform.SetParent(root, false);
        floor.transform.localScale = new Vector3(map.floor.size_x, 0.1f, map.floor.size_z);
        floor.transform.localPosition = new Vector3(map.floor.size_x / 2, -0.05f, map.floor.size_z / 2);

        // Start and goal: coloured squares 2 mm above the floor, markers only (no collider).
        Zone(root, "Start", map.start, new Color(0.1f, 0.75f, 0.2f));
        Zone(root, "Goal", map.goal, new Color(0.9f, 0.15f, 0.1f));

        foreach (var o in map.objects)
        {
            var prefab = Resources.Load<GameObject>("Kenney/" + o.asset);
            if (prefab == null) throw new Exception($"asset not found: Resources/Kenney/{o.asset}");
            var m = new Matrix4x4();
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) m[r, c] = o.matrix[r * 4 + c];
            var go = Instantiate(prefab, root, false);
            go.name = $"{o.id} {o.asset}";
            go.transform.SetLocalPositionAndRotation(m.GetPosition(), m.rotation);
            go.transform.localScale = Vector3.Scale(go.transform.localScale, m.lossyScale);
            var item = go.AddComponent<MapItem>();
            item.id = o.id; item.asset = o.asset; item.footprint = o.footprint;
            // Convex hulls: solid, so a tracked body inside an object still counts as touching it
            // (a non-convex mesh collider is only a surface). Slightly larger than concave shapes.
            foreach (var mf in go.GetComponentsInChildren<MeshFilter>())
            {
                var mc = mf.gameObject.AddComponent<MeshCollider>();
                mc.sharedMesh = mf.sharedMesh;
                mc.convex = true;
            }
            foreach (var t in go.GetComponentsInChildren<Transform>()) t.gameObject.isStatic = true;

            // Rest the object on the floor even if an import changed its pivot.
            var bounds = new Bounds(go.transform.position, Vector3.zero);
            foreach (var rend in go.GetComponentsInChildren<Renderer>()) bounds.Encapsulate(rend.bounds);
            if (Mathf.Abs(bounds.min.y) > 0.01f)
            {
                Debug.LogWarning($"{go.name}: base was at y={bounds.min.y:F3}, moved to the floor");
                go.transform.position -= new Vector3(0, bounds.min.y, 0);
            }
        }
        Physics.SyncTransforms();
        return map;
    }

    static void Zone(Transform root, string name, MapZone zone, Color colour)
    {
        if (zone == null || zone.center == null || zone.center.Length != 2) return;  // older maps have none
        var quad = GameObject.CreatePrimitive(PrimitiveType.Cube);
        quad.name = name;
        DestroyImmediate(quad.GetComponent<Collider>());
        quad.transform.SetParent(root, false);
        quad.transform.localScale = new Vector3(zone.size, 0.004f, zone.size);
        quad.transform.localPosition = new Vector3(zone.center[0], 0.002f, zone.center[1]);
        var renderer = quad.GetComponent<Renderer>();
        renderer.sharedMaterial = new Material(renderer.sharedMaterial) { name = name, color = colour };
    }
}
