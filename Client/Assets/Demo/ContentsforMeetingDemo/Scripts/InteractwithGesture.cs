using System.Collections;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using Microsoft.MixedReality.Toolkit;
using UnityEngine;


public class InteractwithGesture : MonoBehaviour
{
    public Material baseButton;
    public Material highlightButton;

    public Material baseAppButton;
    public Material highlightAppButton;
    
    public PhysicalPressEventRouter pr_yes;
    public RoundedCorners rc_yes;
    public RoundedCorners rc_no;

    
    public PhysicalPressEventRouter pr_0;
    public List<MeshRenderer> mrs;
    

    int curIndex = -1;
    GameObject curButton;
    public GameObject AppSuggestion_buttons;

    public GameObject content_0;
    
    string InputString = "";

    int event_status = 0;
    int tmp = 0;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        /*
            - 각 버튼 object가 활성화되어있을 때 함수 수행

            - sharealert 버튼 2개 찾아서 좌우 제스쳐로 하이라이트
            - 선택 제스쳐 시 DialogManager OnPressYes()

            - Appsuggestions 버튼 3개 찾아서 좌우 제스쳐로 하이라이트
            - 첫 버튼 선택 시 OpenAplication OpenApplicationPrefab(GameObject) 실행

            - 현재 어느 asset이 활성화되어있는지 보고 switch문에 실행할 함수 변경
        */

        switch (event_status)
        {
            case 0:
            {
                Function_A_withString(InputString);
                InputString = "";
                break;
            }
            case 1:
            {
                Function_B_withString(InputString);
                InputString = "";
                break;
            }
            default:
                break;

        }       
    }

    void Function_A_withString(string InputString)
    {
         switch (InputString)
            {
                case "left":
                    InputString = "";
                    curIndex = 0;
                    rc_yes.material = highlightButton;
                    rc_no.material = baseButton;
                    
                    break;
                case "right":
                    InputString = "";
                    curIndex = 1;
                    rc_yes.material = baseButton;
                    rc_no.material = highlightButton;
            
                    break;
                case "tap":
                    if (curIndex == 0)
                    {
                        InputString = "";
                        pr_yes.OnHandPressTriggered();
                        event_status = 1;
                        curIndex = 1;
                    }                    
                    break;
                default:
                    break;
            }        
    }


    void Function_B_withString(string InputString)
    {
         switch (InputString)
            {
                case "left":
                    InputString = "";
                    curIndex -= 1;
                    if (curIndex <= 0)
                        curIndex = 0;
                    
                    tmp = 0;
                    foreach (MeshRenderer mr in mrs) 
                    {
                        if (tmp == curIndex)
                            mr.material = highlightAppButton;
                        else
                            mr.material = baseAppButton;    
                        tmp += 1;
                    }
                    
                    // mr_0.material = highlightAppButton;
                    break;
                case "right":
                    InputString = "";
                    curIndex += 1;
                    if (curIndex >= 2)
                        curIndex = 2;
                        
                    tmp = 0;
                    foreach (MeshRenderer mr in mrs) 
                    {
                        if (tmp == curIndex)
                            mr.material = highlightAppButton;
                        else
                            mr.material = baseAppButton;    
                        tmp += 1;
                    }
                    break;
                case "tap":
                    if (curIndex == 0)
                    {
                        InputString = "";
                        pr_0.OnHandPressTriggered();
                        mrs[0].material = baseAppButton; 
                    }
                    break;
                default:
                    break;
            }        
    }


    public void GetInputMessage(string inputMessage)
    {
        InputString = inputMessage;
    }
}
