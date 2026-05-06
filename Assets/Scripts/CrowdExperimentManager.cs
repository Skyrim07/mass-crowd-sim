using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class CrowdExperimentManager : MonoBehaviour
{
    [Header("Baseline Setup")]
    [SerializeField] private CrowdAgent agentPrefab;
    [SerializeField] private Vector2 spawnAreaSize = new Vector2(50f, 50f);
    [SerializeField, Min(0)] private int agentCount = 100;
    [SerializeField] private int randomSeed = 12345;
    [SerializeField] private bool spawnOnStart = true;

    [Header("NavMesh Sampling")]
    [SerializeField, Min(0.1f)] private float navMeshSampleMaxDistance = 4f;
    [SerializeField, Min(1)] private int navMeshSampleAttempts = 20;

    [Header("Scaling Experiment")]
    [SerializeField] private MetricsLogger metricsLogger;
    [SerializeField] private int[] agentCountsToTest = { 50, 100, 200, 400, 800 };
    [SerializeField, Min(0f)] private float trialDurationSeconds = 30f;

    [Header("Debug UI")]
    [SerializeField] private bool showDebugUi = true;

    private readonly List<CrowdAgent> agents = new List<CrowdAgent>();
    private Coroutine scalingExperimentCoroutine;
    private float smoothedDeltaTime;
    private bool baselineMetricsRunning;

    public IReadOnlyList<CrowdAgent> Agents => agents;
    public int ActiveAgentCount => agents.Count;

    private void Update()
    {
        smoothedDeltaTime += (Time.unscaledDeltaTime - smoothedDeltaTime) * 0.1f;
    }

    private void Start()
    {
        if (spawnOnStart)
        {
            ResetExperiment();
            BeginBaselineMetricsRun();
        }
    }

    private void OnDisable()
    {
        EndBaselineMetricsRun();
    }

    [ContextMenu("Run Scaling Experiment")]
    public void StartScalingExperiment()
    {
        if (scalingExperimentCoroutine != null)
        {
            StopCoroutine(scalingExperimentCoroutine);
        }

        EndBaselineMetricsRun();
        scalingExperimentCoroutine = StartCoroutine(RunScalingExperiment());
    }

    public void ResetExperiment()
    {
        ResetExperiment(agentCount);
    }

    public IEnumerator RunScalingExperiment()
    {
        Random.InitState(randomSeed);

        for (int i = 0; i < agentCountsToTest.Length; i++)
        {
            int testAgentCount = Mathf.Max(0, agentCountsToTest[i]);
            ResetExperiment(testAgentCount);

            if (metricsLogger != null)
            {
                metricsLogger.BeginRun("Baseline", testAgentCount, GetTotalCompletedTasks);
            }

            yield return new WaitForSeconds(trialDurationSeconds);

            if (metricsLogger != null)
            {
                metricsLogger.EndRun();
            }
        }

        scalingExperimentCoroutine = null;
    }

    public bool TryGetRandomDestination(out Vector3 destination)
    {
        return TryGetRandomNavMeshPoint(out destination);
    }

    private void OnGUI()
    {
        if (!showDebugUi)
        {
            return;
        }

        float frameTimeMs = smoothedDeltaTime * 1000f;
        float fps = smoothedDeltaTime > 0f ? 1f / smoothedDeltaTime : 0f;
        string csvPath = metricsLogger != null ? metricsLogger.CsvOutputPath : "No MetricsLogger assigned";

        GUIStyle titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 20,
            fontStyle = FontStyle.Bold
        };

        GUIStyle labelStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 16,
            wordWrap = true
        };

        Rect panelRect = new Rect(10f, 10f, 560f, 275f);
        Rect contentRect = new Rect(24f, 22f, 532f, 250f);

        GUI.Box(panelRect, string.Empty);
        GUILayout.BeginArea(contentRect);
        GUILayout.Label("Crowd Debug", titleStyle);
        GUILayout.Space(6f);
        GUILayout.Label("Mode: Baseline", labelStyle);
        GUILayout.Label($"Agents: {agents.Count}", labelStyle);
        GUILayout.Label($"FPS: {fps:F1}", labelStyle);
        GUILayout.Label($"Frame Time: {frameTimeMs:F2} ms", labelStyle);
        GUILayout.Label($"Stuck Agents: {GetStuckAgentCount()}", labelStyle);
        GUILayout.Label($"Completed Tasks: {GetTotalCompletedTasks()}", labelStyle);
        GUILayout.Label($"CSV: {csvPath}", labelStyle);
        GUILayout.EndArea();
    }

    private void ResetExperiment(int count)
    {
        ClearExistingAgents();

        if (agentPrefab == null)
        {
            Debug.LogError("CrowdExperimentManager requires an agent prefab.");
            return;
        }

        Random.InitState(randomSeed);

        for (int i = 0; i < count; i++)
        {
            if (!TryGetRandomNavMeshPoint(out Vector3 spawnPosition))
            {
                Debug.LogWarning($"Could not find a valid NavMesh spawn point for agent {i}.");
                continue;
            }

            CrowdAgent agent = Instantiate(agentPrefab, spawnPosition, Quaternion.identity, transform);
            agent.name = $"CrowdAgent_{i:0000}";
            agent.Initialize(this);

            if (TryGetRandomDestination(out Vector3 destination))
            {
                agent.SetTarget(destination);
            }

            agents.Add(agent);
        }
    }

    private void BeginBaselineMetricsRun()
    {
        if (metricsLogger == null)
        {
            return;
        }

        metricsLogger.BeginRun("Baseline", agents.Count, GetTotalCompletedTasks);
        baselineMetricsRunning = true;
    }

    private void EndBaselineMetricsRun()
    {
        if (!baselineMetricsRunning || metricsLogger == null)
        {
            return;
        }

        metricsLogger.EndRun();
        baselineMetricsRunning = false;
    }

    private void ClearExistingAgents()
    {
        for (int i = agents.Count - 1; i >= 0; i--)
        {
            if (agents[i] != null)
            {
                Destroy(agents[i].gameObject);
            }
        }

        agents.Clear();
    }

    private bool TryGetRandomNavMeshPoint(out Vector3 point)
    {
        Vector3 areaCenter = transform.position;

        for (int attempt = 0; attempt < navMeshSampleAttempts; attempt++)
        {
            float x = Random.Range(-spawnAreaSize.x * 0.5f, spawnAreaSize.x * 0.5f);
            float z = Random.Range(-spawnAreaSize.y * 0.5f, spawnAreaSize.y * 0.5f);
            Vector3 candidate = areaCenter + new Vector3(x, 0f, z);

            if (NavMesh.SamplePosition(candidate, out NavMeshHit hit, navMeshSampleMaxDistance, NavMesh.AllAreas))
            {
                point = hit.position;
                return true;
            }
        }

        point = areaCenter;
        return false;
    }

    private int GetStuckAgentCount()
    {
        int stuckCount = 0;

        for (int i = 0; i < agents.Count; i++)
        {
            if (agents[i] != null && agents[i].IsStuck)
            {
                stuckCount++;
            }
        }

        return stuckCount;
    }

    private int GetTotalCompletedTasks()
    {
        int completedTasks = 0;

        for (int i = 0; i < agents.Count; i++)
        {
            if (agents[i] != null)
            {
                completedTasks += agents[i].CompletedTasks;
            }
        }

        return completedTasks;
    }
}
