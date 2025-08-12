using Photon.Pun;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class RPC_GlassestoPhone : MonoBehaviour
{
    public static UnityEvent event_onGazeOnObjChange;
    public static UnityEvent event_onIsObjBeingManipChange;
    public static UnityEvent event_onObjClickForWebview;
    void Awake()
    {
        if (event_onGazeOnObjChange == null)
            event_onGazeOnObjChange = new UnityEvent();
        if (event_onIsObjBeingManipChange == null)
            event_onIsObjBeingManipChange = new UnityEvent();
        if (event_onObjClickForWebview == null)
            event_onObjClickForWebview = new UnityEvent();
    }

    [PunRPC]
    void RPC_GazeOnObjChange(string name)
    {
        switch (name)
        {
            case "Antenna":
                DeviceState.Instance.GazeOnObjFromGlasses = GazeOnObjFromGlasses.Antenna;
                break;
            case "Solarpanel":
                DeviceState.Instance.GazeOnObjFromGlasses = GazeOnObjFromGlasses.Solarpanel;
                break;
            case "Headlamp":
                DeviceState.Instance.GazeOnObjFromGlasses = GazeOnObjFromGlasses.Headlamp;
                break;
            case "Null":
                DeviceState.Instance.GazeOnObjFromGlasses = GazeOnObjFromGlasses.Null;
                break;
            default:
                break;
        }
        event_onGazeOnObjChange.Invoke();
    }

    [PunRPC]
    void RPC_IsObjBeingManipChange(bool _bool)
    {
        DeviceState.Instance.IsObjBeingManip = _bool;
        event_onIsObjBeingManipChange.Invoke();
    }

    [PunRPC]
    void RPC_ObjClickForWebview(string name)
    {
        switch (name)
        {
            case "Antenna":
                DeviceState.Instance.SelectedObjFromGlasses = SelectedObjFromGlasses.Antenna;
                break;
            case "Solarpanel":
                DeviceState.Instance.SelectedObjFromGlasses = SelectedObjFromGlasses.Solarpanel;
                break;
            case "Headlamp":
                DeviceState.Instance.SelectedObjFromGlasses = SelectedObjFromGlasses.Headlamp;
                break;
            default:
                break;
        }
        event_onObjClickForWebview.Invoke();
    }
}
