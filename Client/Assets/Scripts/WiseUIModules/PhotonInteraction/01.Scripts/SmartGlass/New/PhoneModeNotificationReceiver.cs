using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PhoneModeNotificationReceiver : MonoBehaviour
{
    [SerializeField]
    private Camera mainCamera;
    [SerializeField]
    private GameObject viewModeOffPrefab;
    [SerializeField]
    private GameObject viewModeOnPrefab;

    private void Awake()
    {
        RPC_PhonetoGlasses.event_veiwModeChange.AddListener(GenViewModeNotification);
    }

    private void GenViewModeNotification()
    {
        GameObject prefabToInstantiate = DeviceState.Instance.ViewMode ? viewModeOnPrefab : viewModeOffPrefab;

        if (mainCamera != null && prefabToInstantiate != null)
        {
            GameObject newObject = Instantiate(prefabToInstantiate);
            newObject.transform.SetParent(mainCamera.transform);
            newObject.transform.localPosition = new Vector3(0, -0.1f, 0.5f);
            newObject.transform.localRotation = Quaternion.Euler(0, 0, 0);
        }
        else
        {
            Debug.LogError("mainCamera 또는 Prefab이 할당되지 않았습니다.");
        }
    }
}
