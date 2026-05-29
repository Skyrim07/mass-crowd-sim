using System;
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
    private string explicitCsvPath;
    private string currentVariant;
    private int currentAgentCount;
    private float elapsedTime;
    private float sampleTimer;
    private float accumulatedFrameTime;
    private int accumulatedFrames;
    private bool isRunning;
    private bool csvInitialized;
    private Func<int> completedTasksProvider;
    private Func<int> stuckAgentsProvider;

    public string CsvOutputPath => string.IsNullOrEmpty(csvPath)
        ? Path.Combine(Application.persistentDataPath, csvFileName)
        : csvPath;

    private void Awake()
    {
        csvPath = ResolveCsvPath();

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

    public void ConfigureCsvOutput(string outputPath, bool resetFile)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            explicitCsvPath = null;
            csvPath = ResolveCsvPath();
        }
        else
        {
            explicitCsvPath = ResolveOutputPath(outputPath);
            csvPath = explicitCsvPath;
            csvFileName = Path.GetFileName(csvPath);
        }

        writeCsvFile = true;
        csvInitialized = false;
        InitializeCsvFile(resetFile);
    }

    public void SetLogToConsole(bool enabled)
    {
        logToConsole = enabled;
    }

    public void SetSampleInterval(float seconds)
    {
        sampleIntervalSeconds = Mathf.Max(0.01f, seconds);
    }

    public void BeginRun(string variantName, int agentCount, Func<int> completedTasksProvider = null, Func<int> stuckAgentsProvider = null)
    {
        currentVariant = variantName;
        currentAgentCount = agentCount;
        this.completedTasksProvider = completedTasksProvider;
        this.stuckAgentsProvider = stuckAgentsProvider;
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
        int completedTasks = completedTasksProvider?.Invoke() ?? 0;
        int stuckAgents = stuckAgentsProvider?.Invoke() ?? 0;
        string line = FormatCsvLine(elapsedTime, currentVariant, currentAgentCount, completedTasks, stuckAgents, averageDeltaTime * 1000f, averageFps);

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

    private void InitializeCsvFile(bool resetFile = false)
    {
        csvPath = ResolveCsvPath();
        string outputDirectory = Path.GetDirectoryName(csvPath);

        if (!string.IsNullOrEmpty(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        if (!resetFile && csvInitialized && File.Exists(csvPath))
        {
            return;
        }

        if (!resetFile && File.Exists(csvPath))
        {
            csvInitialized = true;
            return;
        }

        File.WriteAllText(csvPath, "time_seconds,variant,agent_count,completed_tasks,stuck_agents,average_delta_time_ms,average_fps\n", Encoding.UTF8);
        csvInitialized = true;
    }

    private string ResolveCsvPath()
    {
        return string.IsNullOrEmpty(explicitCsvPath)
            ? Path.Combine(Application.persistentDataPath, csvFileName)
            : explicitCsvPath;
    }

    private static string ResolveOutputPath(string outputPath)
    {
        if (Path.IsPathRooted(outputPath))
        {
            return outputPath;
        }

        string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        return Path.GetFullPath(Path.Combine(projectRoot, outputPath));
    }

    private static string FormatCsvLine(float timeSeconds, string variant, int agentCount, int completedTasks, int stuckAgents, float averageDeltaTimeMs, float averageFps)
    {
        CultureInfo culture = CultureInfo.InvariantCulture;

        return string.Join(",",
            timeSeconds.ToString("F3", culture),
            variant,
            agentCount.ToString(culture),
            completedTasks.ToString(culture),
            stuckAgents.ToString(culture),
            averageDeltaTimeMs.ToString("F3", culture),
            averageFps.ToString("F2", culture));
    }
}
