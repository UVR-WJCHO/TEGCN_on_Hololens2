using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using Microsoft.MixedReality.Toolkit.Utilities;
using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class AlertConfig : MonoBehaviour
{

    [Header("Canvas Elements")]
    private Transform canvas;
    private Image iconImage;
    private Sprite[] iconSheet;
    private Material material;
    private CanvasGroup canvasGroup;
    private TextMeshProUGUI title, description;
    private Transform buttonGroup;
    private Transform[] buttonBackPlates;

    [Header("Notification Settings")]
    [SerializeField] private bool hasButtons;
    [Range(1f, 10f), SerializeField] private float duration = 3.0f;
    private enum StateType { Stationary, Walking }
    private enum IconType
    {
        Settings, Message, Layout, Slack, Folder, Music, Spotify, Camera, Share, Wifi,
        Contacts, Home, SoundOff, SoundOn, Telegram, Keyboard, Phone, ReadMode, ControlMode,
    }

    [SerializeField] private StateType userState;
    [SerializeField] private IconType icon;

    [Header("Notification Text")]
    [TextArea, SerializeField] private string titleText = "Title Text";
    [TextArea, SerializeField] private string descriptionText = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod. ";


    private void Start()
    {
        InitializeComponents();
        SetCanvasAlpha(1);
        SetNotificationContent();
        // StartCoroutine(FadeOut());
    }

    private IEnumerator FadeOut()
    {
        yield return new WaitForSeconds(duration);
        const float fadeDuration = 0.5f;
        float elapsed = 0.0f;
        Color initialColor = material.color;

        while (elapsed < fadeDuration)
        {
            float alphaValue = Mathf.Lerp(1, 0, elapsed / fadeDuration);
            SetCanvasAlpha(alphaValue, initialColor);
            elapsed += Time.deltaTime;
            yield return null;
        }

        SetCanvasAlpha(0);
        Destroy(gameObject);
    }

    private void OnValidate()
    {
        InitializeComponents();
        SetCanvasAlpha(1);
        SetNotificationContent();
    }

    private void InitializeComponents()
    {
        canvas ??= GetComponentInChildren<Canvas>().transform;
        iconImage ??= canvas.Find("Icon").GetComponent<Image>();
        iconSheet ??= Resources.LoadAll<Sprite>("UI/Icon");

        title ??= canvas.Find("Title").GetComponent<TextMeshProUGUI>();
        description ??= canvas.Find("Description").GetComponent<TextMeshProUGUI>();

        material ??= (Material)Resources.Load<Material>("UI/HolographicBackPlateNoBorder");
        canvasGroup ??= canvas.GetComponent<CanvasGroup>();

        buttonGroup ??= canvas.GetComponentInChildren<GridObjectCollection>(true).transform;
        if (buttonBackPlates == null || buttonBackPlates.Length == 0) InitializeButtonBackPlates();
    }

    private void InitializeButtonBackPlates()
    {
        int numButtons = buttonGroup.childCount;
        buttonBackPlates = new Transform[numButtons];
        for (int i = 0; i < numButtons; i++)
            buttonBackPlates[i] = buttonGroup.GetChild(i).GetChild(2).transform;
    }

    private void ConfigureButtons()
    {
        buttonGroup.gameObject.SetActive(hasButtons);
        if (!hasButtons) return;

        var gridObjectCollection = buttonGroup.GetComponent<GridObjectCollection>();

        float cellWidth = userState == StateType.Stationary ? 400 : 260;
        float yPos = userState == StateType.Stationary ? -150 : -180;
        Vector2 sizeDelta = userState == StateType.Stationary ? new Vector2(390, 80) : new Vector2(250, 80);

        gridObjectCollection.CellWidth = cellWidth;
        buttonGroup.localPosition = new Vector3(0, yPos, 0);
        gridObjectCollection.UpdateCollection();

        foreach (var backplates in buttonBackPlates)
        {
            RectTransform rectTransform = backplates.GetComponent<RectTransform>();
            if (rectTransform != null)
                rectTransform.sizeDelta = sizeDelta;
        }

    }

    private void SetCanvasAlpha(float alpha, Color? baseColor = null)
    {
        Color color = baseColor ?? material.color;
        if (canvasGroup != null) canvasGroup.alpha = alpha;
        if (material != null)
        {
            color.a = alpha;
            material.color = color;
        }
    }

    private void SetNotificationContent()
    {
        SetIcon();
        SetPanelLayout();
        SetTextContent();

        ConfigureButtons();
    }

    private void SetIcon()
    {
        if (iconSheet != null && iconImage != null)
            iconImage.sprite = iconSheet[(int)icon];
    }

    private void SetPanelLayout()
    {
        RectTransform rt = canvas.GetComponent<RectTransform>();
        float baseOffset = CalculateBaseOffset(10f, 0.7f);
        float textPosX = title.rectTransform.anchoredPosition.x;
        Vector3 offset = userState == StateType.Stationary ? new Vector3(0, -baseOffset, 0) : new Vector3(baseOffset * 1.48f, 0, 0);
        Vector2 sizeDelta = userState == StateType.Stationary ? new Vector2(800, 200) : new Vector2(520, 260);
        Vector2 titlePos = userState == StateType.Stationary ? new Vector2(textPosX, 42) : new Vector2(textPosX, 57);
        Vector2 descriptionPos = userState == StateType.Stationary ? new Vector2(textPosX, -22) : new Vector2(textPosX, -37);

        AdjustOffset(offset, rt, sizeDelta, titlePos, descriptionPos);
    }

    private void SetTextContent()
    {
        if (title != null) title.text = titleText;
        if (description != null) description.text = descriptionText;
    }

    private void AdjustOffset(Vector3 offset, RectTransform rt, Vector2 sizeDelta, Vector2 titlePos, Vector2 descriptionPos)
    {
        var solverHandler = canvas.GetComponent<SolverHandler>();
        solverHandler.AdditionalOffset = offset;
        rt.sizeDelta = sizeDelta;
        title.rectTransform.anchoredPosition = titlePos;
        description.rectTransform.anchoredPosition = descriptionPos;
    }

    private float CalculateBaseOffset(float angle, float distance) => Mathf.Tan(angle / 2 * Mathf.Deg2Rad) * distance * 2;
}
