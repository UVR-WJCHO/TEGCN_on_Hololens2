using Photon.Pun;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PhoneAttitudeReceiver : MonoBehaviour
{
    [SerializeField]
    private Transform PhoneViz;
    [SerializeField]
    private PhotonView PV;
    private void Awake()
    {
        PV = GameObject.FindGameObjectWithTag("RPCFunction").GetComponent<PhotonView>();
        RPC_PhonetoGlasses.event_syncGyroAttitude.AddListener(UpdatePhoneViz);
    }

    void UpdatePhoneViz()
    {
        PhoneViz.rotation = DeviceState.Instance.GyroAttitude;
    }
}
