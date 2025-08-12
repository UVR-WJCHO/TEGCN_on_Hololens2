using Photon.Pun;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class RPC_PhonetoGlasses : MonoBehaviour
{
    public static UnityEvent event_onObjRoverGenButtonClick;
    public static UnityEvent event_onPhoneSwipeButtonClick;
    public static UnityEvent event_onPhoneGyroButtonClick;
    public static UnityEvent event_onGlassesGyroButtonClick;
    public static UnityEvent event_syncSwipeDelta;
    public static UnityEvent event_syncGyroAttitude;
    public static UnityEvent event_syncGyroDelta;
    public static UnityEvent event_pointerDown;
    public static UnityEvent event_pointerUp;
    public static UnityEvent event_veiwModeChange;


    void Awake()
    {
        if (event_onObjRoverGenButtonClick == null)
            event_onObjRoverGenButtonClick = new UnityEvent();
        if (event_onPhoneSwipeButtonClick == null)
            event_onPhoneSwipeButtonClick = new UnityEvent();
        if (event_onPhoneGyroButtonClick == null)
            event_onPhoneGyroButtonClick = new UnityEvent();
        if (event_onGlassesGyroButtonClick == null)
            event_onGlassesGyroButtonClick = new UnityEvent();

        if (event_syncSwipeDelta == null)
            event_syncSwipeDelta = new UnityEvent();
        if (event_syncGyroAttitude == null)
            event_syncGyroAttitude = new UnityEvent();
        if (event_syncGyroDelta == null)
            event_syncGyroDelta = new UnityEvent();

        if (event_pointerDown == null)
            event_pointerDown = new UnityEvent();
        if (event_pointerUp == null)
            event_pointerUp = new UnityEvent();

        if (event_veiwModeChange == null)
            event_veiwModeChange = new UnityEvent();
    }

    [PunRPC]
    void RPC_GenButtonClick(string name)
    {
        Debug.Log("RPC_GenButtonClick - " + name);
        switch (name)
        {
            case "Antenna":
                DeviceState.Instance.SelectedObjFromPhone = SelectedObjFromPhone.Antenna;
                break;
            case "Solarpanel":
                DeviceState.Instance.SelectedObjFromPhone = SelectedObjFromPhone.Solarpanel;
                break;
            case "Headlamp":
                DeviceState.Instance.SelectedObjFromPhone = SelectedObjFromPhone.Headlamp;
                break;
            default:
                break;
        }
        event_onObjRoverGenButtonClick.Invoke();
    }

    [PunRPC]
    void RPC_PhoneSwipeButtonClick()
    {
        DeviceState.Instance.ObjControlMode = ObjControlMode.PhoneSwipe;
        event_onPhoneSwipeButtonClick.Invoke();
        Debug.Log("RPC_PhoneSwipeButtonClick!");
    }

    [PunRPC]
    void RPC_PhoneGyroButtonClick()
    {
        DeviceState.Instance.ObjControlMode = ObjControlMode.PhoneGyro;
        event_onPhoneGyroButtonClick.Invoke();
        Debug.Log("RPC_PhoneGyroButtonClick!");
    }

    [PunRPC]
    void RPC_GlassesGyroButtonClick()
    {
        DeviceState.Instance.ObjControlMode = ObjControlMode.GlassesGyro;
        event_onGlassesGyroButtonClick.Invoke();
        Debug.Log("RPC_GlassesGyroButtonClick!");
    }

    [PunRPC]
    void RPC_SyncSwipeDelta(Vector3 input)
    {
        DeviceState.Instance.SwipeDelta = input;
        event_syncSwipeDelta.Invoke();
    }

    [PunRPC]
    void RPC_SyncGyroAttitude(Quaternion input)
    {
        DeviceState.Instance.GyroAttitude = input;
        event_syncGyroAttitude.Invoke();
    }


    [PunRPC]
    void RPC_SyncGyroDelta(Vector3 input)
    {
        DeviceState.Instance.GyroDelta = input;
        event_syncGyroDelta.Invoke();
    }

    [PunRPC]
    void RPC_PointerDown()
    {
        event_pointerDown.Invoke();
        //Debug.Log("RPC_PointerDown!");
    }

    [PunRPC]
    void RPC_PointerUp()
    {
        event_pointerUp.Invoke();
        //Debug.Log("RPC_PointerUp!");
    }

    [PunRPC]
    void RPC_ViewModeChange(Boolean viewMode)
    {
        DeviceState.Instance.ViewMode = viewMode;
        event_veiwModeChange.Invoke();
        Debug.Log("RPC_ViewModeChange - " + DeviceState.Instance.ViewMode);
    }
}
