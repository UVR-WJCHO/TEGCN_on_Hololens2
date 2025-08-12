using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities;

public class OpenApplication : MonoBehaviour
{

    public void OpenApplicationPrefab(GameObject obj)
    {
        Transform cam = CameraCache.Main.transform;
        Vector3 pos = cam.position + cam.forward * 0.5f;
        Vector3 dir = cam.forward;
        dir.y = 0f;
        GameObject newObj = Instantiate(obj, pos, Quaternion.LookRotation(dir));
    }
}
