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
    public Boolean ViewMode { get; set; } = false; // false은 평소 모드, true는 읽기 모드
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
        // 로컬 플레이어의 DeviceState를 기존 플레이어의 DeviceState로 동기화
        if (PhotonNetwork.PlayerListOthers.Length > 0)
        {
            Player otherPlayer = PhotonNetwork.PlayerListOthers[0]; // 기존 플레이어 중 한 명 선택
            if (otherPlayer.CustomProperties.TryGetValue("DeviceState", out object deviceStateObj))
            {
                DeviceState otherDeviceState = (DeviceState)deviceStateObj;
                // 기존 플레이어의 DeviceState 정보를 로컬 플레이어의 DeviceState에 복사
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

    //실행중 동기화를 원할시 아래 함수 사용, 현재는 접속시에만 동기화하고 나머지는 RPC 함수로 한번에 적용해서 필요없음
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
