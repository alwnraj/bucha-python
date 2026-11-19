using UnityEngine;

// Marks a spawned map object so contacts can report which one was touched.
public class MapItem : MonoBehaviour
{
    public string id, asset;
    [Tooltip("xmin, zmin, xmax, zmax from the map file (m)")] public float[] footprint;
}
