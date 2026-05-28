using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

public class CrowdTrainingDataLogger : MonoBehaviour
{
    [SerializeField] private bool logToConsoleOnEnd = true;
    [SerializeField] private string csvFileName = "crowd_training_data.csv";
    [SerializeField, Min(1)] private int flushEverySamples = 4096;

    private readonly StringBuilder buffer = new StringBuilder(1024 * 256);
    private string csvPath;
    private string currentTeacher;
    private int currentAgentCount;
    private int bufferedSamples;
    private int totalSamples;
    private bool csvInitialized;

    public bool IsRecording { get; private set; }

    public string CsvOutputPath => string.IsNullOrEmpty(csvPath)
        ? Path.Combine(Application.persistentDataPath, csvFileName)
        : csvPath;

    private void Awake()
    {
        csvPath = Path.Combine(Application.persistentDataPath, csvFileName);
    }

    private void OnDisable()
    {
        EndRun();
    }

    [ContextMenu("Open Training Data Folder")]
    public void OpenTrainingDataFolder()
    {
        string outputPath = CsvOutputPath;
        string outputDirectory = Path.GetDirectoryName(outputPath);

        if (string.IsNullOrEmpty(outputDirectory))
        {
            Debug.LogWarning($"Could not resolve training data folder from path: {outputPath}");
            return;
        }

        Directory.CreateDirectory(outputDirectory);

#if UNITY_EDITOR
        UnityEditor.EditorUtility.RevealInFinder(File.Exists(outputPath) ? outputPath : outputDirectory);
#else
        Application.OpenURL("file:///" + outputDirectory.Replace("\\", "/"));
#endif
    }

    public void BeginRun(string teacherName, int agentCount)
    {
        EndRun();
        InitializeCsvFile();

        currentTeacher = teacherName;
        currentAgentCount = agentCount;
        bufferedSamples = 0;
        totalSamples = 0;
        IsRecording = true;
    }

    public void EndRun()
    {
        if (!IsRecording && bufferedSamples == 0)
        {
            return;
        }

        Flush();
        IsRecording = false;

        if (logToConsoleOnEnd)
        {
            Debug.Log($"Training data samples written: {totalSamples} -> {CsvOutputPath}");
        }
    }

    public void RecordSample(
        float timeSeconds,
        int agentIndex,
        Vector3 position,
        Vector3 targetOffset,
        Vector3 velocity,
        float speed,
        Vector3 nearestNeighborOffset,
        float nearestNeighborDistance,
        int neighborCount,
        Vector2 boundaryDistance,
        Vector3 desiredVelocity)
    {
        if (!IsRecording)
        {
            return;
        }

        CultureInfo culture = CultureInfo.InvariantCulture;
        buffer.Append(timeSeconds.ToString("F4", culture)).Append(',');
        buffer.Append(currentTeacher).Append(',');
        buffer.Append(currentAgentCount.ToString(culture)).Append(',');
        buffer.Append(agentIndex.ToString(culture)).Append(',');
        buffer.Append(position.x.ToString("F4", culture)).Append(',');
        buffer.Append(position.z.ToString("F4", culture)).Append(',');
        buffer.Append(targetOffset.x.ToString("F4", culture)).Append(',');
        buffer.Append(targetOffset.z.ToString("F4", culture)).Append(',');
        buffer.Append(targetOffset.magnitude.ToString("F4", culture)).Append(',');
        buffer.Append(velocity.x.ToString("F4", culture)).Append(',');
        buffer.Append(velocity.z.ToString("F4", culture)).Append(',');
        buffer.Append(speed.ToString("F4", culture)).Append(',');
        buffer.Append(nearestNeighborOffset.x.ToString("F4", culture)).Append(',');
        buffer.Append(nearestNeighborOffset.z.ToString("F4", culture)).Append(',');
        buffer.Append(nearestNeighborDistance.ToString("F4", culture)).Append(',');
        buffer.Append(neighborCount.ToString(culture)).Append(',');
        buffer.Append(boundaryDistance.x.ToString("F4", culture)).Append(',');
        buffer.Append(boundaryDistance.y.ToString("F4", culture)).Append(',');
        buffer.Append(desiredVelocity.x.ToString("F4", culture)).Append(',');
        buffer.Append(desiredVelocity.z.ToString("F4", culture)).Append(',');
        buffer.Append(desiredVelocity.magnitude.ToString("F4", culture)).Append('\n');

        bufferedSamples++;
        totalSamples++;

        if (bufferedSamples >= flushEverySamples)
        {
            Flush();
        }
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

        File.WriteAllText(
            csvPath,
            "time_seconds,teacher,agent_count,agent_index,position_x,position_z,target_offset_x,target_offset_z,target_distance,velocity_x,velocity_z,speed,nearest_neighbor_offset_x,nearest_neighbor_offset_z,nearest_neighbor_distance,neighbor_count,boundary_distance_x,boundary_distance_z,desired_velocity_x,desired_velocity_z,desired_speed\n",
            Encoding.UTF8);
        csvInitialized = true;
    }

    private void Flush()
    {
        if (bufferedSamples == 0)
        {
            return;
        }

        File.AppendAllText(CsvOutputPath, buffer.ToString(), Encoding.UTF8);
        buffer.Length = 0;
        bufferedSamples = 0;
    }
}
