using UnityEngine;
using Photon.Pun;
using Photon.Realtime;
using System;

public enum ObjControlMode
{
    PhoneSwipe,
    PhoneGyro,
    GlassesGyro
}

public enum SelectedObjFromPhone
{
    Antenna,
    Headlamp,
    Solarpanel,
    Null
}

public enum SelectedObjFromGlasses
{
    Antenna,
    Headlamp,
    Solarpanel,
    Null
}

public enum GazeOnObjFromGlasses
{
    Antenna,
    Headlamp,
    Solarpanel,
    Null
}

public class DeviceState : MonoBehaviourPunCallbacks, IPunObservable
{
    public static DeviceState Instance { get; private set; }

    public ObjControlMode ObjControlMode { get; set; } = ObjControlMode.PhoneSwipe;
    public Boolean ViewMode { get; set; } = false; // false�� ��� ���, true�� �б� ���
    public Vector3 SwipeDelta { get; set; }
    public Vector3 GyroDelta { get; set; }
    public Quaternion GyroAttitude { get; set; }
    public bool IsObjBeingManip { get; set; }
    public SelectedObjFromPhone SelectedObjFromPhone { get; set; } = SelectedObjFromPhone.Null;
    public SelectedObjFromGlasses SelectedObjFromGlasses { get; set; } = SelectedObjFromGlasses.Null;
    public GazeOnObjFromGlasses GazeOnObjFromGlasses { get; set; } = GazeOnObjFromGlasses.Null;


    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    public void InitializeFromOtherPlayer()
    {
        // ���� �÷��̾��� DeviceState�� ���� �÷��̾��� DeviceState�� ����ȭ
        if (PhotonNetwork.PlayerListOthers.Length > 0)
        {
            Player otherPlayer = PhotonNetwork.PlayerListOthers[0]; // ���� �÷��̾� �� �� �� ����
            if (otherPlayer.CustomProperties.TryGetValue("DeviceState", out object deviceStateObj))
            {
                DeviceState otherDeviceState = (DeviceState)deviceStateObj;
                // ���� �÷��̾��� DeviceState ������ ���� �÷��̾��� DeviceState�� ����
                ObjControlMode = otherDeviceState.ObjControlMode;
                ViewMode = otherDeviceState.ViewMode;
                SwipeDelta = otherDeviceState.SwipeDelta;
                GyroDelta = otherDeviceState.GyroDelta;
                GyroAttitude = otherDeviceState.GyroAttitude;
                IsObjBeingManip = otherDeviceState.IsObjBeingManip;
                SelectedObjFromPhone = otherDeviceState.SelectedObjFromPhone;
                SelectedObjFromGlasses = otherDeviceState.SelectedObjFromGlasses;
                GazeOnObjFromGlasses = otherDeviceState.GazeOnObjFromGlasses;
            }
        }
    }

    //������ ����ȭ�� ���ҽ� �Ʒ� �Լ� ���, ����� ���ӽÿ��� ����ȭ�ϰ� �������� RPC �Լ��� �ѹ��� �����ؼ� �ʿ����
    public void OnPhotonSerializeView(PhotonStream stream, PhotonMessageInfo info)
    {
        if (stream.IsWriting)
        {
            stream.SendNext((int)ObjControlMode);
            stream.SendNext(ViewMode);
            stream.SendNext(SwipeDelta);
            stream.SendNext(GyroDelta);
            stream.SendNext(GyroAttitude);
            stream.SendNext(IsObjBeingManip);
            stream.SendNext((int)SelectedObjFromPhone);
            stream.SendNext((int)SelectedObjFromGlasses);
            stream.SendNext((int)GazeOnObjFromGlasses);
        }
        else
        {
            ObjControlMode = (ObjControlMode)(int)stream.ReceiveNext();
            ViewMode = (Boolean)stream.ReceiveNext();
            SwipeDelta = (Vector3)stream.ReceiveNext();
            GyroDelta = (Vector3)stream.ReceiveNext();
            GyroAttitude = (Quaternion)stream.ReceiveNext();
            IsObjBeingManip = (bool)stream.ReceiveNext();
            SelectedObjFromPhone = (SelectedObjFromPhone)(int)stream.ReceiveNext();
            SelectedObjFromGlasses = (SelectedObjFromGlasses)(int)stream.ReceiveNext();
            GazeOnObjFromGlasses = (GazeOnObjFromGlasses)(int)stream.ReceiveNext();
        }
    }
}
