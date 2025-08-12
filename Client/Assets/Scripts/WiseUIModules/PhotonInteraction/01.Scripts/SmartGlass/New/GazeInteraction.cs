using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using Photon.Pun;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARSubsystems;

public class GazeInteraction : MonoBehaviour
{

    [SerializeField]
    private PhotonView PV;
    [SerializeField]
    private Transform mainCameraTransform;
    private Quaternion mainCameraRot;
    private Transform mainCameraTransformProcessed;

    public GameObject antennaPrefab;
    public GameObject headlampPrefab;
    public GameObject solarpanelPrefab;

    public float gyroPositionGain;
    public float swipePositionGain;
    public float headPositionGain;

    private IMixedRealityEyeGazeProvider eyeTrackingProvider = null;
    private IMixedRealityEyeGazeProvider EyeTrackingProvider => eyeTrackingProvider ?? (eyeTrackingProvider = CoreServices.InputSystem?.EyeGazeProvider);
    private GameObject gazeOnObject;
    private GameObject manipObject;

    private float rotY;
    private float rotX;
    private float prerotY;
    private float prerotX;
    private float delRotY;
    private float delRotX;

    private Ray ray;
    private RaycastHit hitData;

    private GameObject newObject;
    private float scale = 0.3f;

    public GameObject GazeOnObject
    {
        get { return gazeOnObject; }
        set
        {
            if (gazeOnObject != value)
            {
                PV.RPC("RPC_GazeOnObjChange", RpcTarget.All, gazeOnObject?.name);
                gazeOnObject = value;
            }
        }
    }

    public GameObject ManipObject
    {
        get { return manipObject; }
        set { manipObject = value; }
    }

    private void Awake()
    {
        PV = GameObject.FindGameObjectWithTag("RPCFunction").GetComponent<PhotonView>();
        mainCameraTransform = GameObject.FindGameObjectWithTag("MainCamera").GetComponent<Transform>();
        gazeOnObject = null;
        manipObject = null;
        mainCameraTransformProcessed = new GameObject("ProcessedCameraTransform").transform;
        mainCameraTransformProcessed.position = Vector3.zero;
        mainCameraTransformProcessed.rotation = Quaternion.identity;
        mainCameraTransformProcessed.localScale = Vector3.one;
        mainCameraTransformProcessed.parent = mainCameraTransform.parent;
        RPC_PhonetoGlasses.event_onObjRoverGenButtonClick.AddListener(GenerateWithGaze);
        RPC_PhonetoGlasses.event_pointerDown.AddListener(IsMoveCheck);
        RPC_PhonetoGlasses.event_pointerUp.AddListener(IsMoveFalse);
    }


    void Update()
    {
        //정보: gaze가 인식되는 최소 거리가 존재함
        gazeOnObject = EyeTrackingProvider.HitInfo.transform?.gameObject;

        //Debug.Log("gazeOnObject : " + gazeOnObject?.name);
        //Debug.Log("manipObject : " + manipObject);
        //Debug.Log("hitPosition" + EyeTrackingProvider.HitPosition);

        //FireInteractionSpaceRay();

        if (manipObject != null)
        {
            //Debug.Log(manipObject);
            switch (DeviceState.Instance.ObjControlMode)
            {
                case ObjControlMode.PhoneSwipe:
                    MoveWithSwipe();
                    break;
                case ObjControlMode.PhoneGyro:
                    MoveWithGyro();
                    break;
                case ObjControlMode.GlassesGyro:
                    MoveWithHead();
                    break;
            }
        }
    }

    private void GenerateWithGaze()
    {
        Debug.Log("GenerateWithGaze - " + DeviceState.Instance.SelectedObjFromPhone);
        //if (gazeOnObject?.layer == LayerMask.NameToLayer("Spatial Awareness"))
        //{
        if (manipObject == null)
        {
            switch (DeviceState.Instance.SelectedObjFromPhone)
            {
                case SelectedObjFromPhone.Antenna:
                    Debug.Log("gen Antenna");
                    newObject = Instantiate(antennaPrefab, EyeTrackingProvider.HitPosition + new Vector3(0, antennaPrefab.transform.localScale.y * scale / 3, 0), Quaternion.Euler(new Vector3(0, 0, 0)));
                    newObject.transform.localScale = new Vector3(scale, scale, scale);
                    break;
                case SelectedObjFromPhone.Headlamp:
                    Debug.Log("gen Headlamp");
                    newObject = Instantiate(headlampPrefab, EyeTrackingProvider.HitPosition + new Vector3(0, headlampPrefab.transform.localScale.y * scale / 3, 0), Quaternion.Euler(new Vector3(0, 0, 0)));
                    newObject.transform.localScale = new Vector3(scale, scale, scale);
                    break;
                case SelectedObjFromPhone.Solarpanel:
                    Debug.Log("gen Solarpanel");
                    newObject = Instantiate(solarpanelPrefab, EyeTrackingProvider.HitPosition + new Vector3(0, solarpanelPrefab.transform.localScale.y * scale / 3, 0), Quaternion.Euler(new Vector3(0, 0, 0)));
                    newObject.transform.localScale = new Vector3(scale, scale, scale);
                    break;
                default:
                    break;
            }
        }
        //}
        //else
        //{
        //    Debug.Log(EyeTrackingProvider.HitInfo.transform);
        //}
    }

    private void MoveWithGyro()
    {
        manipObject.transform.Translate(RotDelXZtoMainCameraCordinate(DeviceState.Instance.GyroDelta * gyroPositionGain), Space.World);
    }

    private void MoveWithSwipe()
    {
        Debug.Log("MoveWithSwipe");
        manipObject.transform.Translate(RotDelXZtoMainCameraCordinate(DeviceState.Instance.SwipeDelta * swipePositionGain), Space.World);
    }

    private void MoveWithHead()
    {
        HeadRotDelYXUpdate();
        manipObject.transform.Translate(RotDelXZtoMainCameraCordinate(RotDelYXtoScrDelXZ() * headPositionGain), Space.World);
    }

    private void HeadRotDelYXUpdate()
    {
        rotY = CameraCache.Main.transform.rotation.y;
        rotX = CameraCache.Main.transform.rotation.x;
        delRotY = rotY - prerotY;
        delRotX = rotX - prerotX;
        prerotY = rotY;
        prerotX = rotX;
    }

    private Vector3 RotDelYXtoScrDelXZ()
    {
        return new Vector3(delRotY * Time.deltaTime, 0, -delRotX * Time.deltaTime);
    }

    private Vector3 RotDelXZtoMainCameraCordinate(Vector3 DelXZ)
    {
        mainCameraRot = mainCameraTransform.rotation;
        mainCameraRot = Quaternion.Euler(0f, mainCameraRot.eulerAngles.y, 0f);
        mainCameraTransformProcessed.rotation = mainCameraRot;
        return mainCameraTransformProcessed.TransformDirection(DelXZ);
    }

    private void IsMoveCheck()
    {
        Debug.Log("IsMoveCheck");
        if (GazeOnObject != null)
        {
            if (GazeOnObject.tag == "Movable")
            {
                manipObject = gazeOnObject;
                PV.RPC("RPC_IsObjBeingManipChange", RpcTarget.All, true);
                //Debug.Log("IsMoveCheck : " + manipObject);
            }
            else
            {
                //Debug.Log("Wall");
            }
        }
    }

    private void IsMoveFalse()
    {
        manipObject = null;
        PV.RPC("RPC_IsObjBeingManipChange", RpcTarget.All, false);
        Debug.Log("IsMoveFalse : " + manipObject);
    }
}
