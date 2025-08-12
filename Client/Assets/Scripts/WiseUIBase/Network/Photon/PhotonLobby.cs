using Photon.Pun;
using Photon.Realtime;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PhotonLobby : MonoBehaviourPunCallbacks
{
    public static PhotonLobby lobby;
    [SerializeField]
    private DeviceState deviceState;

    private void Awake()
    {
        lobby = this; //Creates the singleton, lives withing the Main menu scene.
        deviceState = GameObject.FindGameObjectWithTag("NetworkManager").GetComponent<DeviceState>();
    }

    void Start()
    {
        PhotonNetwork.ConnectUsingSettings(); //Connects to Master photon server.
    }

    public override void OnConnectedToMaster() //Called when app connected to Master photon server
    {
        PhotonNetwork.JoinRandomRoom();
        PhotonNetwork.AutomaticallySyncScene = true;
    }

    public override void OnJoinedRoom()
    {
        base.OnJoinedRoom();
        Debug.Log("Joined room!");
        Debug.Log(PhotonNetwork.CurrentRoom.PlayerCount);

        if (PhotonNetwork.IsConnected && PhotonNetwork.CurrentRoom.PlayerCount > 1)
        {
            // 현재 방에 두 명 이상의 플레이어(기기)가 있다면, 공유하고있는 값으로 동기화
            deviceState.InitializeFromOtherPlayer();
        }
    }

    public override void OnJoinRandomFailed(short returnCode, string message) //Called when failed to join room (no room)
    {
        CreateRoom();
    }

    void CreateRoom()
    {
        int randomRoomName = Random.Range(0, 10);
        RoomOptions roomOps = new RoomOptions() { IsVisible = true, IsOpen = true, MaxPlayers = 10 };
        PhotonNetwork.CreateRoom("Room" + randomRoomName, roomOps);
    }

    public override void OnCreateRoomFailed(short returnCode, string message) //Called when failed to create room (name duplicate)
    {
        CreateRoom();
    }
}
