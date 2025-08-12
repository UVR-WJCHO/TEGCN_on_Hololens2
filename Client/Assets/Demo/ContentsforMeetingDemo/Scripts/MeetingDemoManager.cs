using System.Collections;
using Microsoft.MixedReality.Toolkit.UI;
using UnityEngine;

public class MeetingDemoManager : MonoBehaviour
{
    // Reference to GameObjects A and B
    public GameObject objectA;
    public GameObject objectB;

    // State variable
    private bool isCountdownActive = false;

    // Method to start the countdown
    public void OnCountdownStart()
    {
        if (!isCountdownActive)
        {
            isCountdownActive = true;
            StartCoroutine(ActivateObjectsAfterDelay(5f));
        }
    }

    // Coroutine to activate the objects after a delay
    private IEnumerator ActivateObjectsAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay); // Wait for the specified time
        if (objectA != null) objectA.SetActive(true);
        if (objectB != null) objectB.SetActive(true);
        isCountdownActive = false; // Reset the state after activation
    }
}
