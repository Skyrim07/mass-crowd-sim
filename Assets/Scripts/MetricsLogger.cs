using System.Globalization;
using System.IO;
using System.Text;
using SKCell;
using UnityEngine;

public class MetricsLogger : MonoBehaviour
{
    [SerializeField] private bool logToConsole = true;
    [SerializeField] private bool writeCsvFile = false;
    [SerializeField] private string csvFileName = "crowd_metrics.csv";
    [SerializeField, Min(0.01f)] private float sampleIntervalSeconds = 1f;

    private string csvPath;
    private string currentVariant;
    private int currentAgentCount;
    private float elapsedTime;
    private float sampleTimer;
    private float accumulatedFrameTime;
    private int accumulatedFrames;
    private bool isRunning;
    private bool csvInitialized;

    public string CsvOutputPath => string.IsNullOrEmpty(csvPath)
        ? Path.Combine(Application.persistentDataPath, csvFileName)
        : csvPath;

    private void Awake()
    {
        csvPath = Path.Combine(Application.persistentDataPath, csvFileName);

        if (writeCsvFile)
        {
            InitializeCsvFile();
        }
    }

    [SKInspectorButton("Open CSV Folder")]
    public void OpenCsvFolder()
    {
        string outputPath = CsvOutputPath;
        string outputDirectory = Path.GetDirectoryName(outputPath);

        if (string.IsNullOrEmpty(outputDirectory))
        {
            Debug.LogWarning($"Could not resolve CSV output folder from path: {outputPath}");
            return;
        }

        Directory.CreateDirectory(outputDirectory);

#if UNITY_EDITOR
        UnityEditor.EditorUtility.RevealInFinder(File.Exists(outputPath) ? outputPath : outputDirectory);
#else
        Application.OpenURL("file:///" + outputDirectory.Replace("\\", "/"));
#endif
    }

    private void Update()
    {
        if (!isRunning)
        {
            return;
        }

        RecordFrame(Time.deltaTime);
    }

    public void BeginRun(string variantName, int agentCount)
    {
        currentVariant = variantName;
        currentAgentCount = agentCount;
        elapsedTime = 0f;
        sampleTimer = 0f;
        accumulatedFrameTime = 0f;
        accumulatedFrames = 0;
        isRunning = true;

        if (writeCsvFile)
        {
            InitializeCsvFile();
        }
    }

    public void BeginRun(ExperimentConfig experimentConfig)
    {
        BeginRun(experimentConfig.variant.ToString(), experimentConfig.agentCount);
    }

    public void EndRun()
    {
        if (!isRunning)
        {
            return;
        }

        isRunning = false;

        if (writeCsvFile && !string.IsNullOrEmpty(csvPath))
        {
            Debug.Log($"Metrics written to {csvPath}");
        }
    }

    private void RecordFrame(float deltaTime)
    {
        elapsedTime += deltaTime;
        sampleTimer += deltaTime;
        accumulatedFrameTime += deltaTime;
        accumulatedFrames++;

        if (sampleTimer < sampleIntervalSeconds)
        {
            return;
        }

        float averageDeltaTime = accumulatedFrameTime / Mathf.Max(1, accumulatedFrames);
        float averageFps = 1f / Mathf.Max(averageDeltaTime, 0.0001f);
        string line = FormatCsvLine(elapsedTime, currentVariant, currentAgentCount, averageDeltaTime * 1000f, averageFps);

        if (logToConsole)
        {
            Debug.Log(line);
        }

        if (writeCsvFile && !string.IsNullOrEmpty(csvPath))
        {
            File.AppendAllText(csvPath, line + "\n", Encoding.UTF8);
        }

        sampleTimer = 0f;
        accumulatedFrameTime = 0f;
        accumulatedFrames = 0;
    }

    private void InitializeCsvFile()
    {
        csvPath = Path.Combine(Application.persistentDataPath, csvFileName);
        string outputDirectory = Path.GetDirectoryName(csvPath);

        if (!string.IsNullOrEmpty(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        if (csvInitialized && File.Exists(csvPath))
        {
            return;
        }

        File.WriteAllText(csvPath, "time_seconds,variant,agent_count,average_delta_time_ms,average_fps\n", Encoding.UTF8);
        csvInitialized = true;
    }

    private static string FormatCsvLine(float timeSeconds, string variant, int agentCount, float averageDeltaTimeMs, float averageFps)
    {
        CultureInfo culture = CultureInfo.InvariantCulture;

        return string.Join(",",
            timeSeconds.ToString("F3", culture),
            variant,
            agentCount.ToString(culture),
            averageDeltaTimeMs.ToString("F3", culture),
            averageFps.ToString("F2", culture));
    }
}
