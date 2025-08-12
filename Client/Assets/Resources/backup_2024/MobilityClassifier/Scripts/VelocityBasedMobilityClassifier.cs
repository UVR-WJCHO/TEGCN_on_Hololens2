using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities;
using TMPro;

public class VelocityBasedMobilityClassifier : MonoBehaviour
{
    public GameObject statePlate;
    public MobilityState mobilityState = MobilityState.Stationary;

    [SerializeField]
    private float velocity;
    [SerializeField]
    private float threshold = 0.5f;

    private Vector3 speed;
    private Vector3 prev_position;
    private float prev_timestamp_sec;

    private GameObject mainCamera;
    
    public void Start()
    {
        mainCamera = Camera.main.gameObject;

        if (statePlate == null)
        {
            statePlate = Instantiate(Resources.Load("MobilityStatePlate") as GameObject);
            statePlate.transform.parent = mainCamera.transform;
            statePlate.transform.localPosition = new Vector3(-0.05f, 0.05f, 0.3f);
            statePlate.transform.localRotation = Quaternion.Euler(0, 0, 0);
        }


    }
    public void SetActiveMobileStatePlate(bool flag)
    {
        statePlate.SetActive(flag);
    }

    public void DetectMobilityState(float curretTimestamp_sec)
    {
        speed = (mainCamera.transform.position - prev_position) / (curretTimestamp_sec- prev_timestamp_sec);
        
        velocity = speed.magnitude;

        if (velocity > threshold)
        {
            mobilityState = MobilityState.Walking;
            statePlate.transform.Find("Text").GetComponent<TextMeshPro>().text = "Context recognized:\r\nwalking";
        }
        else
        {
            mobilityState = MobilityState.Stationary;
            statePlate.transform.Find("Text").GetComponent<TextMeshPro>().text = "Context recognized:\r\nstationary";
        }
        prev_position = mainCamera.transform.position;
        prev_timestamp_sec = curretTimestamp_sec;
    }

}
