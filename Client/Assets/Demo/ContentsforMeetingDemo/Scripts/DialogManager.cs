using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities;

public class DialogManager : MonoBehaviour
{
    public GameObject obj;

    public void OnPressYes()
    {
        Transform cam = CameraCache.Main.transform;
        Vector3 pos = cam.position + cam.forward * 1f - cam.up * 0.5f;
        Vector3 dir = cam.forward;
        dir.y = 0f;
        GameObject newObj = Instantiate(obj, pos, Quaternion.LookRotation(dir));
        this.gameObject.SetActive(false);
    }

    public void OnPressNo()
    {
        this.gameObject.SetActive(false);
    }

}
