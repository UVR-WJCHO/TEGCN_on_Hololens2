using System.Collections;
using UnityEngine;

public class ActivateObjectAfterDelay : MonoBehaviour
{
    // The object to be activated
    public GameObject targetObject;

    // Delay in seconds
    public float delayInSeconds = 1f;

    // Public method to start the activation
    public void Start()
    {
        if (targetObject != null)
        {
            StartCoroutine(ActivateObjectCoroutine());
        }
        else
        {
            Debug.LogWarning("Target object is not assigned.");
        }
    }

    // Coroutine to activate the object after the delay
    private IEnumerator ActivateObjectCoroutine()
    {
        yield return new WaitForSeconds(delayInSeconds);
        targetObject.SetActive(true);
        Debug.Log($"Object '{targetObject.name}' activated after {delayInSeconds} seconds.");
    }
}
