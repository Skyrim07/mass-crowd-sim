using System.Collections;
using System.Collections.Generic;
using System.IO;
using Unity.Barracuda;
using Unity.Collections;
using Unity.Entities;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.Rendering;
using Unity.Transforms;

public class CrowdExperimentManager : MonoBehaviour
{
    public enum SimulationAlgorithm
    {
        BaselineNavMesh,
        SpatialHashGpuInstanced,
        SpatialHashTeacherTrainingData,
        LearnedPolicyGpuInstanced,
        DotsEcsGpuInstanced,
        DotsEcsBehaviorLodGpuInstanced,
        EcsLearnedPolicyGpuInstanced
    }

    private struct SimAgent
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 target;
        public float lowSpeedTimer;
        public bool isStuck;
        public int completedTasks;
    }

    private struct InstancedRenderPart
    {
        public Mesh mesh;
        public Material[] materials;
        public Matrix4x4 localMatrix;
        public ShadowCastingMode shadowCastingMode;
        public bool receiveShadows;
        public int layer;
    }

    private struct NeighborObservation
    {
        public Vector3 separation;
        public Vector3 nearestOffset;
        public float nearestDistance;
        public int neighborCount;
    }

    private const int MaxInstancesPerDrawCall = 1023;
    private const int LearnedPolicyFeatureCount = 12;

    [Header("Algorithm")]
    [SerializeField] private SimulationAlgorithm simulationAlgorithm = SimulationAlgorithm.BaselineNavMesh;

    [Header("Baseline Setup")]
    [SerializeField] private CrowdAgent agentPrefab;
    [SerializeField] private Vector2 spawnAreaSize = new Vector2(50f, 50f);
    [SerializeField, Min(0)] private int agentCount = 100;
    [SerializeField] private int randomSeed = 12345;
    [SerializeField] private bool spawnOnStart = true;

    [Header("Baseline NavMesh Sampling")]
    [SerializeField, Min(0.1f)] private float navMeshSampleMaxDistance = 4f;
    [SerializeField, Min(1)] private int navMeshSampleAttempts = 20;

    [Header("Scaling Experiment")]
    [SerializeField] private MetricsLogger metricsLogger;
    [SerializeField] private int[] agentCountsToTest = { 50, 100, 200, 400, 800 };
    [SerializeField, Min(0f)] private float trialDurationSeconds = 30f;

    [Header("Full Experiment Batch")]
    [SerializeField] private SimulationAlgorithm[] batchAlgorithmsToTest =
    {
        SimulationAlgorithm.BaselineNavMesh,
        SimulationAlgorithm.SpatialHashGpuInstanced,
        SimulationAlgorithm.SpatialHashTeacherTrainingData,
        SimulationAlgorithm.LearnedPolicyGpuInstanced,
        SimulationAlgorithm.DotsEcsGpuInstanced,
        SimulationAlgorithm.DotsEcsBehaviorLodGpuInstanced,
        SimulationAlgorithm.EcsLearnedPolicyGpuInstanced
    };
    [SerializeField] private int[] batchAgentCountsToTest = { 1000, 2000, 5000, 10000 };
    [SerializeField, Min(0f)] private float batchWarmupSeconds = 2f;
    [SerializeField, Min(0f)] private float batchTrialDurationSeconds = 30f;
    [SerializeField, Min(0.01f)] private float batchMetricsSampleInterval = 5f;
    [SerializeField] private string batchCsvOutputPath = "Assets/Data/experiment_batch/crowd_experiment_metrics.csv";
    [SerializeField] private bool resetBatchCsvOnStart = true;
    [SerializeField] private bool batchLogSamplesToConsole = false;
    [SerializeField] private bool scaleSpawnAreaForBatch = true;
    [SerializeField, Min(0.001f)] private float batchSpawnAreaOccupancyRatio = 0.05f;
    [SerializeField] private bool quitApplicationAfterBatch = false;

    [Header("AI Training Data")]
    [SerializeField] private CrowdTrainingDataLogger trainingDataLogger;
    [SerializeField] private bool collectSpatialTeacherTrainingData = false;
    [SerializeField, Min(1)] private int maxTrainingSamplesPerFrame = 256;
    [SerializeField, Min(1)] private int trainingSampleFrameInterval = 2;

    [Header("Learned Policy")]
    [SerializeField] private NNModel learnedPolicyModelAsset;
    [SerializeField] private WorkerFactory.Type learnedPolicyWorkerType = WorkerFactory.Type.Auto;
    [SerializeField, Min(1)] private int learnedPolicyBatchSize = 512;
    [SerializeField, Range(0f, 1f)] private float learnedPolicyVelocityBlend = 1f;

    [Header("Spatial Hash + GPU Instancing")]
    [SerializeField] private Mesh instancedAgentMesh;
    [SerializeField] private Material instancedAgentMaterial;
    [SerializeField, Min(0.01f)] private float instancedAgentScale = 1f;
    [SerializeField, Min(0.01f)] private float instancedAgentSpeed = 3.5f;
    [SerializeField, Min(0.01f)] private float instancedTurnResponsiveness = 8f;
    [SerializeField, Min(0.1f)] private float spatialCellSize = 2.5f;
    [SerializeField, Min(0.1f)] private float neighborRadius = 2f;
    [SerializeField, Min(0f)] private float separationStrength = 2.5f;
    [SerializeField, Min(0.01f)] private float instancedTargetReachedDistance = 1.25f;
    [SerializeField, Min(0f)] private float instancedStuckSpeedThreshold = 0.1f;
    [SerializeField, Min(0f)] private float instancedStuckTimeThreshold = 2f;

    [Header("ECS Behavior LOD")]
    [SerializeField, Min(0f)] private float ecsLodNearDistance = 12f;
    [SerializeField, Min(0f)] private float ecsLodMidDistance = 24f;
    [SerializeField, Min(0f)] private float ecsLodFarDistance = 38f;
    [SerializeField, Min(1)] private int ecsLodNearTickInterval = 1;
    [SerializeField, Min(1)] private int ecsLodMidTickInterval = 2;
    [SerializeField, Min(1)] private int ecsLodFarTickInterval = 4;
    [SerializeField, Min(1)] private int ecsLodVeryFarTickInterval = 8;
    [SerializeField, Range(0f, 1f)] private float ecsLodMidSeparationScale = 0.65f;
    [SerializeField, Range(0f, 1f)] private float ecsLodFarSeparationScale = 0.25f;
    [SerializeField, Range(0f, 1f)] private float ecsLodVeryFarSeparationScale = 0f;

    [Header("Debug UI")]
    [SerializeField] private bool showDebugUi = true;

    private readonly List<CrowdAgent> agents = new List<CrowdAgent>();
    private readonly Dictionary<Vector2Int, List<int>> spatialGrid = new Dictionary<Vector2Int, List<int>>();
    private readonly List<InstancedRenderPart> instancedRenderParts = new List<InstancedRenderPart>();
    private readonly Dictionary<Material, Material> runtimeInstancedMaterials = new Dictionary<Material, Material>();
    private readonly Matrix4x4[] instanceMatrices = new Matrix4x4[MaxInstancesPerDrawCall];
    private Coroutine scalingExperimentCoroutine;
    private SimAgent[] simAgents;
    private EntityQuery ecsAgentQuery;
    private EntityQuery ecsSettingsQuery;
    private bool ecsQueriesInitialized;
    private float nextEcsMetricSampleTime;
    private int cachedEcsAgentCount;
    private int cachedEcsStuckCount;
    private int cachedEcsCompletedTasks;
    private int trainingSampleCursor;
    private int trainingFrameCounter;
    private Model learnedPolicyModel;
    private IWorker learnedPolicyWorker;
    private Vector3[] learnedPolicyDesiredVelocities;
    private float[] learnedPolicyInputBuffer;
    private bool learnedPolicyWarningShown;
    private float smoothedDeltaTime;
    private bool metricsRunActive;

    public IReadOnlyList<CrowdAgent> Agents => agents;
    public int ActiveAgentCount => simulationAlgorithm == SimulationAlgorithm.BaselineNavMesh
        ? agents.Count
        : IsSpatialAlgorithm(simulationAlgorithm)
            ? simAgents?.Length ?? 0
            : GetEcsAgentCount();

    private void Update()
    {
        smoothedDeltaTime += (Time.unscaledDeltaTime - smoothedDeltaTime) * 0.1f;

        if (IsSpatialAlgorithm(simulationAlgorithm))
        {
            UpdateSpatialHashSimulation(Time.deltaTime);
        }

        if (simulationAlgorithm == SimulationAlgorithm.EcsLearnedPolicyGpuInstanced)
        {
            EvaluateEcsLearnedPolicy();
        }
    }

    private void LateUpdate()
    {
        if (IsSpatialAlgorithm(simulationAlgorithm))
        {
            RenderSpatialHashAgents();
        }

        if (IsEcsAlgorithm(simulationAlgorithm))
        {
            RenderEcsAgents();
        }
    }

    private void Start()
    {
        if (spawnOnStart)
        {
            ScaleSpawnArea(.05f);
            ResetExperiment();
            BeginMetricsRun();
        }
    }

    private void OnDisable()
    {
        EndMetricsRun();
        ReleaseRuntimeInstancedMaterials();
        ReleaseLearnedPolicyWorker();
        ClearEcsAgents();
    }

    public void ScaleSpawnArea(float ratio)
    {
        float agentRadius = instancedAgentScale;

        if (agentPrefab != null && agentPrefab.TryGetComponent(out NavMeshAgent navMeshAgent))
        {
            agentRadius = navMeshAgent.radius * instancedAgentScale;
        }

        float totalAgentArea = agentCount * Mathf.PI * agentRadius * agentRadius;
        float newArea = totalAgentArea / ratio;
        float newLength = Mathf.Sqrt(newArea);
        spawnAreaSize = new Vector2(newLength, newLength);
    }

    [ContextMenu("Run Scaling Experiment")]
    public void StartScalingExperiment()
    {
        if (scalingExperimentCoroutine != null)
        {
            StopCoroutine(scalingExperimentCoroutine);
        }

        EndMetricsRun();
        scalingExperimentCoroutine = StartCoroutine(RunScalingExperiment());
    }

    [ContextMenu("Run Full Experiment Batch")]
    public void StartFullExperimentBatch()
    {
        if (scalingExperimentCoroutine != null)
        {
            StopCoroutine(scalingExperimentCoroutine);
        }

        EndMetricsRun();
        scalingExperimentCoroutine = StartCoroutine(RunFullExperimentBatch());
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
                metricsLogger.BeginRun(simulationAlgorithm.ToString(), testAgentCount, GetTotalCompletedTasks, GetStuckAgentCount);
            }

            yield return new WaitForSeconds(trialDurationSeconds);

            if (metricsLogger != null)
            {
                metricsLogger.EndRun();
            }
        }

        scalingExperimentCoroutine = null;
    }

    public IEnumerator RunFullExperimentBatch()
    {
        PrepareBatchMetricsLogger();

        SimulationAlgorithm[] algorithms = batchAlgorithmsToTest;
        if (algorithms == null || algorithms.Length == 0)
        {
            algorithms = (SimulationAlgorithm[])global::System.Enum.GetValues(typeof(SimulationAlgorithm));
        }

        int[] counts = batchAgentCountsToTest;
        if (counts == null || counts.Length == 0)
        {
            counts = new[] { agentCount };
        }

        Debug.Log($"Starting full crowd experiment batch. CSV: {metricsLogger.CsvOutputPath}");

        for (int algorithmIndex = 0; algorithmIndex < algorithms.Length; algorithmIndex++)
        {
            simulationAlgorithm = algorithms[algorithmIndex];

            for (int countIndex = 0; countIndex < counts.Length; countIndex++)
            {
                int testAgentCount = Mathf.Max(0, counts[countIndex]);
                agentCount = testAgentCount;

                if (scaleSpawnAreaForBatch)
                {
                    ScaleSpawnArea(batchSpawnAreaOccupancyRatio);
                }

                Debug.Log($"Running {simulationAlgorithm} with {testAgentCount} agents.");
                ResetExperiment(testAgentCount);

                if (batchWarmupSeconds > 0f)
                {
                    yield return new WaitForSeconds(batchWarmupSeconds);
                }
                else
                {
                    yield return null;
                }

                metricsLogger.BeginRun(simulationAlgorithm.ToString(), ActiveAgentCount, GetTotalCompletedTasks, GetStuckAgentCount);
                metricsRunActive = true;
                BeginTrainingDataRun();

                if (batchTrialDurationSeconds > 0f)
                {
                    yield return new WaitForSeconds(batchTrialDurationSeconds);
                }
                else
                {
                    yield return null;
                }

                EndMetricsRun();
            }
        }

        scalingExperimentCoroutine = null;
        Debug.Log($"Finished full crowd experiment batch. CSV: {metricsLogger.CsvOutputPath}");

        if (quitApplicationAfterBatch)
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }

    public bool TryGetRandomDestination(out Vector3 destination)
    {
        if (simulationAlgorithm == SimulationAlgorithm.BaselineNavMesh)
        {
            return TryGetRandomNavMeshPoint(out destination);
        }

        destination = GetRandomSimulationPoint();
        return true;
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

        Rect panelRect = new Rect(10f, 10f, 720f, 345f);
        Rect contentRect = new Rect(24f, 22f, 692f, 320f);

        GUI.Box(panelRect, string.Empty);
        GUILayout.BeginArea(contentRect);
        GUILayout.Label("Crowd Debug", titleStyle);
        GUILayout.Space(6f);
        GUILayout.Label($"Mode: {simulationAlgorithm}", labelStyle);
        GUILayout.Label($"Agents: {ActiveAgentCount}", labelStyle);
        GUILayout.Label($"FPS: {fps:F1}", labelStyle);
        GUILayout.Label($"Frame Time: {frameTimeMs:F2} ms", labelStyle);
        GUILayout.Label($"Stuck Agents: {GetStuckAgentCount()}", labelStyle);
        GUILayout.Label($"Completed Tasks: {GetTotalCompletedTasks()}", labelStyle);
        GUILayout.Label($"CSV: {csvPath}", labelStyle);
        GUILayout.Space(8f);

        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Baseline"))
        {
            SwitchAlgorithm(SimulationAlgorithm.BaselineNavMesh);
        }

        if (GUILayout.Button("Spatial GPU"))
        {
            SwitchAlgorithm(SimulationAlgorithm.SpatialHashGpuInstanced);
        }

        if (GUILayout.Button("AI Data"))
        {
            SwitchAlgorithm(SimulationAlgorithm.SpatialHashTeacherTrainingData);
        }

        if (GUILayout.Button("Learned"))
        {
            SwitchAlgorithm(SimulationAlgorithm.LearnedPolicyGpuInstanced);
        }

        GUILayout.EndHorizontal();
        GUILayout.BeginHorizontal();

        if (GUILayout.Button("DOTS ECS"))
        {
            SwitchAlgorithm(SimulationAlgorithm.DotsEcsGpuInstanced);
        }

        if (GUILayout.Button("ECS LOD"))
        {
            SwitchAlgorithm(SimulationAlgorithm.DotsEcsBehaviorLodGpuInstanced);
        }

        if (GUILayout.Button("ECS Learned"))
        {
            SwitchAlgorithm(SimulationAlgorithm.EcsLearnedPolicyGpuInstanced);
        }

        if (GUILayout.Button("Reset"))
        {
            RestartCurrentRun();
        }

        GUILayout.EndHorizontal();
        GUILayout.EndArea();
    }

    private void ResetExperiment(int count)
    {
        ClearExistingAgents();
        ClearSpatialAgents();
        ClearEcsAgents();

        Random.InitState(randomSeed);

        if (IsEcsAlgorithm(simulationAlgorithm))
        {
            InitializeEcsAgents(
                count,
                simulationAlgorithm == SimulationAlgorithm.DotsEcsBehaviorLodGpuInstanced,
                simulationAlgorithm == SimulationAlgorithm.EcsLearnedPolicyGpuInstanced);
            return;
        }

        if (IsSpatialAlgorithm(simulationAlgorithm))
        {
            InitializeSpatialHashAgents(count);
            return;
        }

        if (agentPrefab == null)
        {
            Debug.LogError("CrowdExperimentManager requires an agent prefab.");
            return;
        }

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

    private void SwitchAlgorithm(SimulationAlgorithm nextAlgorithm)
    {
        if (simulationAlgorithm == nextAlgorithm)
        {
            return;
        }

        simulationAlgorithm = nextAlgorithm;
        RestartCurrentRun();
    }

    private void RestartCurrentRun()
    {
        if (scalingExperimentCoroutine != null)
        {
            StopCoroutine(scalingExperimentCoroutine);
            scalingExperimentCoroutine = null;
        }

        EndMetricsRun();
        ResetExperiment(agentCount);
        BeginMetricsRun();
    }

    private static bool IsEcsAlgorithm(SimulationAlgorithm algorithm)
    {
        return algorithm == SimulationAlgorithm.DotsEcsGpuInstanced
            || algorithm == SimulationAlgorithm.DotsEcsBehaviorLodGpuInstanced
            || algorithm == SimulationAlgorithm.EcsLearnedPolicyGpuInstanced;
    }

    private static bool IsSpatialAlgorithm(SimulationAlgorithm algorithm)
    {
        return algorithm == SimulationAlgorithm.SpatialHashGpuInstanced
            || algorithm == SimulationAlgorithm.SpatialHashTeacherTrainingData
            || algorithm == SimulationAlgorithm.LearnedPolicyGpuInstanced;
    }

    private static bool IsLearnedPolicyAlgorithm(SimulationAlgorithm algorithm)
    {
        return algorithm == SimulationAlgorithm.LearnedPolicyGpuInstanced
            || algorithm == SimulationAlgorithm.EcsLearnedPolicyGpuInstanced;
    }

    private void BeginMetricsRun()
    {
        if (metricsLogger == null)
        {
            BeginTrainingDataRun();
            return;
        }

        metricsLogger.BeginRun(simulationAlgorithm.ToString(), ActiveAgentCount, GetTotalCompletedTasks, GetStuckAgentCount);
        metricsRunActive = true;
        BeginTrainingDataRun();
    }

    private void PrepareBatchMetricsLogger()
    {
        if (metricsLogger == null)
        {
            metricsLogger = GetComponent<MetricsLogger>();

            if (metricsLogger == null)
            {
                metricsLogger = gameObject.AddComponent<MetricsLogger>();
            }
        }

        metricsLogger.ConfigureCsvOutput(batchCsvOutputPath, resetBatchCsvOnStart);
        metricsLogger.SetSampleInterval(batchMetricsSampleInterval);
        metricsLogger.SetLogToConsole(batchLogSamplesToConsole);

        if (!string.IsNullOrWhiteSpace(batchCsvOutputPath))
        {
            string outputDirectory = Path.GetDirectoryName(metricsLogger.CsvOutputPath);

            if (!string.IsNullOrEmpty(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }
        }
    }

    private void EndMetricsRun()
    {
        EndTrainingDataRun();

        if (!metricsRunActive || metricsLogger == null)
        {
            return;
        }

        metricsLogger.EndRun();
        metricsRunActive = false;
    }

    private void BeginTrainingDataRun()
    {
        trainingSampleCursor = 0;
        trainingFrameCounter = 0;

        if ((collectSpatialTeacherTrainingData || simulationAlgorithm == SimulationAlgorithm.SpatialHashTeacherTrainingData)
            && trainingDataLogger == null)
        {
            trainingDataLogger = GetComponent<CrowdTrainingDataLogger>();

            if (trainingDataLogger == null)
            {
                trainingDataLogger = gameObject.AddComponent<CrowdTrainingDataLogger>();
            }
        }

        if (!ShouldCollectSpatialTrainingData())
        {
            trainingDataLogger?.EndRun();
            return;
        }

        trainingDataLogger.BeginRun(simulationAlgorithm.ToString(), ActiveAgentCount);
    }

    private void EndTrainingDataRun()
    {
        trainingDataLogger?.EndRun();
    }

    private bool ShouldCollectSpatialTrainingData()
    {
        return (collectSpatialTeacherTrainingData || simulationAlgorithm == SimulationAlgorithm.SpatialHashTeacherTrainingData)
            && trainingDataLogger != null
            && IsSpatialAlgorithm(simulationAlgorithm);
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

    private void ClearSpatialAgents()
    {
        simAgents = null;
        spatialGrid.Clear();
    }

    private void ClearEcsAgents()
    {
        if (!TryGetEntityManager(out EntityManager entityManager))
        {
            return;
        }

        EnsureEcsQueries(entityManager);

        if (ecsQueriesInitialized)
        {
            entityManager.DestroyEntity(ecsAgentQuery);
            entityManager.DestroyEntity(ecsSettingsQuery);
            ecsAgentQuery = default;
            ecsSettingsQuery = default;
            ecsQueriesInitialized = false;
        }

        cachedEcsAgentCount = 0;
        cachedEcsStuckCount = 0;
        cachedEcsCompletedTasks = 0;
        nextEcsMetricSampleTime = 0f;
    }

    private void InitializeEcsAgents(int count, bool enableBehaviorLod, bool enableLearnedPolicy)
    {
        if (!TryResolveInstancedRenderingAssets())
        {
            Debug.LogError("DotsEcsGpuInstanced requires an agent mesh and material. Assign them directly or keep a renderable agent prefab assigned.");
            return;
        }

        if (!TryGetEntityManager(out EntityManager entityManager))
        {
            Debug.LogError("No default ECS world is available. DOTS packages may still be importing.");
            return;
        }

        EnsureEcsQueries(entityManager);

        float lodNearDistance = ecsLodNearDistance;
        float lodMidDistance = Mathf.Max(lodNearDistance, ecsLodMidDistance);
        float lodFarDistance = Mathf.Max(lodMidDistance, ecsLodFarDistance);

        Entity settingsEntity = entityManager.CreateEntity(typeof(CrowdEcsSettings));
        entityManager.SetComponentData(settingsEntity, new CrowdEcsSettings
        {
            Center = ToFloat3(transform.position),
            SpawnAreaSize = new Unity.Mathematics.float2(spawnAreaSize.x, spawnAreaSize.y),
            AgentSpeed = instancedAgentSpeed,
            TurnResponsiveness = instancedTurnResponsiveness,
            SpatialCellSize = spatialCellSize,
            NeighborRadius = neighborRadius,
            SeparationStrength = separationStrength,
            TargetReachedDistance = instancedTargetReachedDistance,
            StuckSpeedThreshold = instancedStuckSpeedThreshold,
            StuckTimeThreshold = instancedStuckTimeThreshold,
            EnableLearnedPolicy = enableLearnedPolicy ? 1 : 0,
            EnableBehaviorLod = enableBehaviorLod ? 1 : 0,
            LodNearDistance = lodNearDistance,
            LodMidDistance = lodMidDistance,
            LodFarDistance = lodFarDistance,
            LodNearTickInterval = Mathf.Max(1, ecsLodNearTickInterval),
            LodMidTickInterval = Mathf.Max(1, ecsLodMidTickInterval),
            LodFarTickInterval = Mathf.Max(1, ecsLodFarTickInterval),
            LodVeryFarTickInterval = Mathf.Max(1, ecsLodVeryFarTickInterval),
            LodMidSeparationScale = ecsLodMidSeparationScale,
            LodFarSeparationScale = ecsLodFarSeparationScale,
            LodVeryFarSeparationScale = ecsLodVeryFarSeparationScale
        });

        EntityArchetype agentArchetype = entityManager.CreateArchetype(
            typeof(CrowdEcsAgent),
            typeof(CrowdEcsRandom),
            typeof(Translation),
            typeof(Rotation),
            typeof(LocalToWorld));

        int safeCount = Mathf.Max(0, count);
        cachedEcsAgentCount = safeCount;
        cachedEcsStuckCount = 0;
        cachedEcsCompletedTasks = 0;
        nextEcsMetricSampleTime = 0f;

        for (int i = 0; i < safeCount; i++)
        {
            Vector3 spawnPosition = GetRandomSimulationPoint();
            Vector3 destination = GetRandomSimulationPoint();

            uint randomStateSeed = (uint)Mathf.Max(1, randomSeed + i + 1);
            Unity.Mathematics.Random randomState = new Unity.Mathematics.Random(randomStateSeed);
            Vector3 initialVelocity = Random.insideUnitSphere.WithY(0f).normalized * instancedAgentSpeed;

            Entity agentEntity = entityManager.CreateEntity(agentArchetype);
            entityManager.SetComponentData(agentEntity, new CrowdEcsAgent
            {
                Velocity = ToFloat3(initialVelocity),
                Target = ToFloat3(destination),
                LowSpeedTimer = 0f,
                NextThinkTick = enableBehaviorLod ? Random.Range(0, Mathf.Max(1, ecsLodVeryFarTickInterval)) : 0,
                LodLevel = 0,
                IsStuck = 0,
                CompletedTasks = 0
            });

            entityManager.SetComponentData(agentEntity, new CrowdEcsRandom { Value = randomState });
            entityManager.SetComponentData(agentEntity, new Translation { Value = ToFloat3(spawnPosition) });
            entityManager.SetComponentData(agentEntity, new Rotation { Value = Unity.Mathematics.quaternion.identity });
        }
    }

    private void InitializeSpatialHashAgents(int count)
    {
        if (!TryResolveInstancedRenderingAssets())
        {
            Debug.LogError("SpatialHashGpuInstanced requires an agent mesh and material. Assign them directly or keep a renderable agent prefab assigned.");
            return;
        }

        simAgents = new SimAgent[Mathf.Max(0, count)];

        for (int i = 0; i < simAgents.Length; i++)
        {
            Vector3 spawnPosition = GetRandomSimulationPoint();
            Vector3 destination = GetRandomSimulationPoint();

            simAgents[i] = new SimAgent
            {
                position = spawnPosition,
                velocity = Random.insideUnitSphere.WithY(0f).normalized * instancedAgentSpeed,
                target = destination,
                lowSpeedTimer = 0f,
                isStuck = false,
                completedTasks = 0
            };
        }
    }

    private bool TryResolveInstancedRenderingAssets()
    {
        if (instancedRenderParts.Count > 0)
        {
            return true;
        }

        instancedRenderParts.Clear();

        if (instancedAgentMesh != null && instancedAgentMaterial != null)
        {
            instancedRenderParts.Add(new InstancedRenderPart
            {
                mesh = instancedAgentMesh,
                materials = new[] { instancedAgentMaterial },
                localMatrix = Matrix4x4.identity,
                shadowCastingMode = ShadowCastingMode.On,
                receiveShadows = true,
                layer = gameObject.layer
            });

            return true;
        }

        if (agentPrefab == null)
        {
            return false;
        }

        Transform root = agentPrefab.transform;
        MeshFilter[] meshFilters = agentPrefab.GetComponentsInChildren<MeshFilter>(true);

        for (int i = 0; i < meshFilters.Length; i++)
        {
            MeshFilter meshFilter = meshFilters[i];
            MeshRenderer meshRenderer = meshFilter.GetComponent<MeshRenderer>();

            if (meshRenderer == null || !meshRenderer.enabled || meshFilter.sharedMesh == null)
            {
                continue;
            }

            Material[] materials = meshRenderer.sharedMaterials;

            if (materials == null || materials.Length == 0)
            {
                continue;
            }

            instancedRenderParts.Add(new InstancedRenderPart
            {
                mesh = meshFilter.sharedMesh,
                materials = materials,
                localMatrix = root.worldToLocalMatrix * meshFilter.transform.localToWorldMatrix,
                shadowCastingMode = meshRenderer.shadowCastingMode,
                receiveShadows = meshRenderer.receiveShadows,
                layer = meshRenderer.gameObject.layer
            });
        }

        return instancedRenderParts.Count > 0;
    }

    private void UpdateSpatialHashSimulation(float deltaTime)
    {
        if (simAgents == null || simAgents.Length == 0)
        {
            return;
        }

        BuildSpatialGrid();
        bool collectTrainingDataThisFrame = ShouldRecordTrainingDataThisFrame();
        bool useLearnedPolicy = IsLearnedPolicyAlgorithm(simulationAlgorithm)
            && TryEvaluateLearnedPolicy(neighborRadius * neighborRadius, Mathf.Max(1, Mathf.CeilToInt(neighborRadius / spatialCellSize)));
        int recordedTrainingSamples = 0;

        float neighborRadiusSqr = neighborRadius * neighborRadius;
        float targetReachedDistanceSqr = instancedTargetReachedDistance * instancedTargetReachedDistance;
        float steeringBlend = 1f - Mathf.Exp(-instancedTurnResponsiveness * deltaTime);
        int cellSearchRadius = Mathf.Max(1, Mathf.CeilToInt(neighborRadius / spatialCellSize));

        for (int i = 0; i < simAgents.Length; i++)
        {
            SimAgent agent = simAgents[i];
            Vector3 toTarget = (agent.target - agent.position).WithY(0f);

            if (toTarget.sqrMagnitude <= targetReachedDistanceSqr)
            {
                agent.completedTasks++;

                if (TryGetRandomDestination(out Vector3 nextTarget))
                {
                    agent.target = nextTarget;
                    toTarget = (agent.target - agent.position).WithY(0f);
                }
            }

            Vector3 desiredVelocity = toTarget.sqrMagnitude > 0.0001f
                ? toTarget.normalized * instancedAgentSpeed
                : Vector3.zero;

            NeighborObservation neighborObservation = CalculateNeighborObservation(i, agent.position, neighborRadiusSqr, cellSearchRadius);
            if (useLearnedPolicy)
            {
                desiredVelocity = Vector3.Lerp(desiredVelocity, learnedPolicyDesiredVelocities[i], learnedPolicyVelocityBlend);
            }
            else
            {
                desiredVelocity += neighborObservation.separation * separationStrength;
            }

            if (desiredVelocity.sqrMagnitude > instancedAgentSpeed * instancedAgentSpeed)
            {
                desiredVelocity = desiredVelocity.normalized * instancedAgentSpeed;
            }

            if (collectTrainingDataThisFrame && ShouldRecordTrainingSample(i, ref recordedTrainingSamples))
            {
                trainingDataLogger.RecordSample(
                    Time.time,
                    i,
                    agent.position,
                    toTarget,
                    agent.velocity,
                    agent.velocity.magnitude,
                    neighborObservation.nearestOffset,
                    neighborObservation.nearestDistance,
                    neighborObservation.neighborCount,
                    GetBoundaryDistance(agent.position),
                    desiredVelocity);
            }

            Vector3 previousPosition = agent.position;
            agent.velocity = Vector3.Lerp(agent.velocity, desiredVelocity, steeringBlend);
            agent.position += agent.velocity * deltaTime;
            agent.position = ClampToSpawnArea(agent.position);

            float currentSpeed = deltaTime > 0f
                ? (agent.position - previousPosition).magnitude / deltaTime
                : 0f;

            bool hasTarget = (agent.target - agent.position).WithY(0f).sqrMagnitude > targetReachedDistanceSqr;
            bool movingTooSlowly = currentSpeed < instancedStuckSpeedThreshold;

            if (hasTarget && movingTooSlowly)
            {
                agent.lowSpeedTimer += deltaTime;
                agent.isStuck = agent.lowSpeedTimer > instancedStuckTimeThreshold;
            }
            else
            {
                agent.lowSpeedTimer = 0f;
                agent.isStuck = false;
            }

            simAgents[i] = agent;
        }

        if (collectTrainingDataThisFrame)
        {
            trainingFrameCounter = 0;
        }
    }

    private bool ShouldRecordTrainingDataThisFrame()
    {
        if (!ShouldCollectSpatialTrainingData() || !trainingDataLogger.IsRecording)
        {
            return false;
        }

        trainingFrameCounter++;
        return trainingFrameCounter >= trainingSampleFrameInterval;
    }

    private bool ShouldRecordTrainingSample(int agentIndex, ref int recordedSamples)
    {
        if (recordedSamples >= maxTrainingSamplesPerFrame)
        {
            return false;
        }

        if (agentIndex != trainingSampleCursor)
        {
            return false;
        }

        recordedSamples++;
        trainingSampleCursor = (trainingSampleCursor + 1) % Mathf.Max(1, simAgents.Length);

        return true;
    }

    private bool TryEvaluateLearnedPolicy(float neighborRadiusSqr, int cellSearchRadius)
    {
        if (!TryPrepareLearnedPolicy())
        {
            return false;
        }

        EnsureLearnedPolicyBuffers(simAgents.Length);

        int batchCapacity = Mathf.Max(1, learnedPolicyBatchSize);
        int agentIndex = 0;

        while (agentIndex < simAgents.Length)
        {
            int batchCount = Mathf.Min(batchCapacity, simAgents.Length - agentIndex);

            for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
            {
                int currentAgentIndex = agentIndex + batchIndex;
                SimAgent agent = simAgents[currentAgentIndex];
                NeighborObservation neighborObservation = CalculateNeighborObservation(
                    currentAgentIndex,
                    agent.position,
                    neighborRadiusSqr,
                    cellSearchRadius);

                WriteLearnedPolicyFeatures(batchIndex, agent, neighborObservation);
            }

            ExecuteLearnedPolicyBatch(batchCount, agentIndex);

            agentIndex += batchCount;
        }

        return true;
    }

    private void EvaluateEcsLearnedPolicy()
    {
        if (!TryGetEntityManager(out EntityManager entityManager) || !TryPrepareLearnedPolicy())
        {
            return;
        }

        EnsureEcsQueries(entityManager);

        int agentCount = ecsAgentQuery.CalculateEntityCount();
        if (agentCount == 0)
        {
            return;
        }

        EnsureLearnedPolicyBuffers(agentCount);

        NativeArray<Entity> entities = ecsAgentQuery.ToEntityArray(Allocator.TempJob);
        NativeArray<CrowdEcsAgent> ecsAgents = ecsAgentQuery.ToComponentDataArray<CrowdEcsAgent>(Allocator.TempJob);
        NativeArray<Translation> translations = ecsAgentQuery.ToComponentDataArray<Translation>(Allocator.TempJob);

        try
        {
            BuildEcsSpatialGrid(translations);
            float neighborRadiusSqr = neighborRadius * neighborRadius;
            int cellSearchRadius = Mathf.Max(1, Mathf.CeilToInt(neighborRadius / spatialCellSize));
            float steeringBlend = 1f - Mathf.Exp(-instancedTurnResponsiveness * Time.deltaTime);
            int batchCapacity = Mathf.Max(1, learnedPolicyBatchSize);
            int agentIndex = 0;

            while (agentIndex < agentCount)
            {
                int batchCount = Mathf.Min(batchCapacity, agentCount - agentIndex);

                for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
                {
                    int currentAgentIndex = agentIndex + batchIndex;
                    CrowdEcsAgent agent = ecsAgents[currentAgentIndex];
                    Vector3 position = ToVector3(translations[currentAgentIndex].Value);
                    NeighborObservation neighborObservation = CalculateEcsNeighborObservation(
                        currentAgentIndex,
                        position,
                        translations,
                        neighborRadiusSqr,
                        cellSearchRadius);

                    WriteLearnedPolicyFeatures(batchIndex, agent, position, neighborObservation);
                }

                ExecuteLearnedPolicyBatch(batchCount, agentIndex);

                for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
                {
                    int currentAgentIndex = agentIndex + batchIndex;
                    CrowdEcsAgent agent = ecsAgents[currentAgentIndex];
                    Vector3 position = ToVector3(translations[currentAgentIndex].Value);
                    Vector3 toTarget = (ToVector3(agent.Target) - position).WithY(0f);
                    Vector3 targetVelocity = toTarget.sqrMagnitude > 0.0001f
                        ? toTarget.normalized * instancedAgentSpeed
                        : Vector3.zero;

                    Vector3 predictedVelocity = Vector3.Lerp(
                        targetVelocity,
                        learnedPolicyDesiredVelocities[currentAgentIndex],
                        learnedPolicyVelocityBlend);

                    if (predictedVelocity.sqrMagnitude > instancedAgentSpeed * instancedAgentSpeed)
                    {
                        predictedVelocity = predictedVelocity.normalized * instancedAgentSpeed;
                    }

                    Vector3 blendedVelocity = Vector3.Lerp(ToVector3(agent.Velocity), predictedVelocity, steeringBlend);
                    agent.Velocity = ToFloat3(blendedVelocity);
                    ecsAgents[currentAgentIndex] = agent;
                    entityManager.SetComponentData(entities[currentAgentIndex], agent);
                }

                agentIndex += batchCount;
            }
        }
        finally
        {
            translations.Dispose();
            ecsAgents.Dispose();
            entities.Dispose();
        }
    }

    private void ExecuteLearnedPolicyBatch(int batchCount, int destinationOffset)
    {
        int inputCount = batchCount * LearnedPolicyFeatureCount;
        float[] tensorInput = learnedPolicyInputBuffer;

        if (inputCount != learnedPolicyInputBuffer.Length)
        {
            tensorInput = new float[inputCount];
            System.Array.Copy(learnedPolicyInputBuffer, tensorInput, inputCount);
        }

        using (Tensor input = new Tensor(batchCount, 1, 1, LearnedPolicyFeatureCount, tensorInput))
        {
            learnedPolicyWorker.Execute(input);
            Tensor output = learnedPolicyWorker.PeekOutput();

            for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
            {
                float desiredX = output[batchIndex, 0, 0, 0];
                float desiredZ = output[batchIndex, 0, 0, 1];
                Vector3 desiredVelocity = new Vector3(desiredX, 0f, desiredZ);

                if (float.IsNaN(desiredVelocity.x) || float.IsInfinity(desiredVelocity.x)
                    || float.IsNaN(desiredVelocity.z) || float.IsInfinity(desiredVelocity.z))
                {
                    desiredVelocity = Vector3.zero;
                }

                learnedPolicyDesiredVelocities[destinationOffset + batchIndex] = desiredVelocity;
            }
        }
    }

    private bool TryPrepareLearnedPolicy()
    {
        if (learnedPolicyWorker != null)
        {
            return true;
        }

        if (learnedPolicyModelAsset == null)
        {
            learnedPolicyModelAsset = TryLoadDefaultLearnedPolicyAsset();
        }

        if (learnedPolicyModelAsset == null)
        {
            if (!learnedPolicyWarningShown)
            {
                Debug.LogWarning("Learned policy experiments require a Barracuda NNModel. Import Assets/Models/crowd_policy.onnx and assign it to Learned Policy Model Asset.");
                learnedPolicyWarningShown = true;
            }

            return false;
        }

        learnedPolicyModel = ModelLoader.Load(learnedPolicyModelAsset);
        learnedPolicyWorker = WorkerFactory.CreateWorker(learnedPolicyWorkerType, learnedPolicyModel);
        learnedPolicyWarningShown = false;
        return true;
    }

    private NNModel TryLoadDefaultLearnedPolicyAsset()
    {
#if UNITY_EDITOR
        return UnityEditor.AssetDatabase.LoadAssetAtPath<NNModel>("Assets/Models/crowd_policy.onnx");
#else
        return null;
#endif
    }

    private void EnsureLearnedPolicyBuffers(int agentCount)
    {
        if (learnedPolicyDesiredVelocities == null || learnedPolicyDesiredVelocities.Length < agentCount)
        {
            learnedPolicyDesiredVelocities = new Vector3[agentCount];
        }

        int inputCount = Mathf.Max(1, learnedPolicyBatchSize) * LearnedPolicyFeatureCount;

        if (learnedPolicyInputBuffer == null || learnedPolicyInputBuffer.Length < inputCount)
        {
            learnedPolicyInputBuffer = new float[inputCount];
        }
    }

    private void WriteLearnedPolicyFeatures(int batchIndex, SimAgent agent, NeighborObservation neighborObservation)
    {
        Vector3 targetOffset = (agent.target - agent.position).WithY(0f);
        Vector2 boundaryDistance = GetBoundaryDistance(agent.position);
        WriteLearnedPolicyFeatures(batchIndex, targetOffset, agent.velocity, boundaryDistance, neighborObservation);
    }

    private void WriteLearnedPolicyFeatures(int batchIndex, CrowdEcsAgent agent, Vector3 position, NeighborObservation neighborObservation)
    {
        Vector3 targetOffset = (ToVector3(agent.Target) - position).WithY(0f);
        Vector3 velocity = ToVector3(agent.Velocity);
        Vector2 boundaryDistance = GetBoundaryDistance(position);
        WriteLearnedPolicyFeatures(batchIndex, targetOffset, velocity, boundaryDistance, neighborObservation);
    }

    private void WriteLearnedPolicyFeatures(
        int batchIndex,
        Vector3 targetOffset,
        Vector3 velocity,
        Vector2 boundaryDistance,
        NeighborObservation neighborObservation)
    {
        int featureOffset = batchIndex * LearnedPolicyFeatureCount;

        learnedPolicyInputBuffer[featureOffset] = targetOffset.x;
        learnedPolicyInputBuffer[featureOffset + 1] = targetOffset.z;
        learnedPolicyInputBuffer[featureOffset + 2] = targetOffset.magnitude;
        learnedPolicyInputBuffer[featureOffset + 3] = velocity.x;
        learnedPolicyInputBuffer[featureOffset + 4] = velocity.z;
        learnedPolicyInputBuffer[featureOffset + 5] = velocity.magnitude;
        learnedPolicyInputBuffer[featureOffset + 6] = neighborObservation.nearestOffset.x;
        learnedPolicyInputBuffer[featureOffset + 7] = neighborObservation.nearestOffset.z;
        learnedPolicyInputBuffer[featureOffset + 8] = neighborObservation.nearestDistance;
        learnedPolicyInputBuffer[featureOffset + 9] = neighborObservation.neighborCount;
        learnedPolicyInputBuffer[featureOffset + 10] = boundaryDistance.x;
        learnedPolicyInputBuffer[featureOffset + 11] = boundaryDistance.y;
    }

    private void ReleaseLearnedPolicyWorker()
    {
        if (learnedPolicyWorker == null)
        {
            return;
        }

        learnedPolicyWorker.Dispose();
        learnedPolicyWorker = null;
        learnedPolicyModel = null;
    }

    private void BuildSpatialGrid()
    {
        foreach (List<int> cellAgents in spatialGrid.Values)
        {
            cellAgents.Clear();
        }

        for (int i = 0; i < simAgents.Length; i++)
        {
            Vector2Int cell = GetSpatialCell(simAgents[i].position);

            if (!spatialGrid.TryGetValue(cell, out List<int> cellAgents))
            {
                cellAgents = new List<int>();
                spatialGrid.Add(cell, cellAgents);
            }

            cellAgents.Add(i);
        }
    }

    private void BuildEcsSpatialGrid(NativeArray<Translation> translations)
    {
        foreach (List<int> cellAgents in spatialGrid.Values)
        {
            cellAgents.Clear();
        }

        for (int i = 0; i < translations.Length; i++)
        {
            Vector2Int cell = GetSpatialCell(ToVector3(translations[i].Value));

            if (!spatialGrid.TryGetValue(cell, out List<int> cellAgents))
            {
                cellAgents = new List<int>();
                spatialGrid.Add(cell, cellAgents);
            }

            cellAgents.Add(i);
        }
    }

    private NeighborObservation CalculateNeighborObservation(int agentIndex, Vector3 position, float neighborRadiusSqr, int cellSearchRadius)
    {
        Vector2Int centerCell = GetSpatialCell(position);
        NeighborObservation observation = new NeighborObservation
        {
            nearestDistance = Mathf.Sqrt(neighborRadiusSqr)
        };

        for (int xOffset = -cellSearchRadius; xOffset <= cellSearchRadius; xOffset++)
        {
            for (int yOffset = -cellSearchRadius; yOffset <= cellSearchRadius; yOffset++)
            {
                Vector2Int neighborCell = new Vector2Int(centerCell.x + xOffset, centerCell.y + yOffset);

                if (!spatialGrid.TryGetValue(neighborCell, out List<int> cellAgents))
                {
                    continue;
                }

                for (int i = 0; i < cellAgents.Count; i++)
                {
                    int otherIndex = cellAgents[i];

                    if (otherIndex == agentIndex)
                    {
                        continue;
                    }

                    Vector3 offset = (position - simAgents[otherIndex].position).WithY(0f);
                    float distanceSqr = offset.sqrMagnitude;
                    float d4 = distanceSqr * distanceSqr;

                    if (distanceSqr <= 0.0001f || distanceSqr > neighborRadiusSqr)
                    {
                        continue;
                    }

                    float distance = Mathf.Sqrt(distanceSqr);
                    // observation.separation += offset / distance * (1f - distance / neighborRadius);
                    observation.separation += offset * (1.0f / (d4 + 0.01f));
                    observation.neighborCount++;

                    if (distance < observation.nearestDistance)
                    {
                        observation.nearestDistance = distance;
                        observation.nearestOffset = offset;
                    }
                }
            }
        }

        return observation;
    }

    private NeighborObservation CalculateEcsNeighborObservation(
        int agentIndex,
        Vector3 position,
        NativeArray<Translation> translations,
        float neighborRadiusSqr,
        int cellSearchRadius)
    {
        Vector2Int centerCell = GetSpatialCell(position);
        NeighborObservation observation = new NeighborObservation
        {
            nearestDistance = Mathf.Sqrt(neighborRadiusSqr)
        };

        for (int xOffset = -cellSearchRadius; xOffset <= cellSearchRadius; xOffset++)
        {
            for (int yOffset = -cellSearchRadius; yOffset <= cellSearchRadius; yOffset++)
            {
                Vector2Int neighborCell = new Vector2Int(centerCell.x + xOffset, centerCell.y + yOffset);

                if (!spatialGrid.TryGetValue(neighborCell, out List<int> cellAgents))
                {
                    continue;
                }

                for (int i = 0; i < cellAgents.Count; i++)
                {
                    int otherIndex = cellAgents[i];

                    if (otherIndex == agentIndex)
                    {
                        continue;
                    }

                    Vector3 offset = (position - ToVector3(translations[otherIndex].Value)).WithY(0f);
                    float distanceSqr = offset.sqrMagnitude;
                    float d4 = distanceSqr * distanceSqr;

                    if (distanceSqr <= 0.0001f || distanceSqr > neighborRadiusSqr)
                    {
                        continue;
                    }

                    float distance = Mathf.Sqrt(distanceSqr);
                    // observation.separation += offset / distance * (1f - distance / neighborRadius);
                    observation.separation += offset * (1.0f / (d4 + 0.01f));
                    observation.neighborCount++;

                    if (distance < observation.nearestDistance)
                    {
                        observation.nearestDistance = distance;
                        observation.nearestOffset = offset;
                    }
                }
            }
        }

        return observation;
    }

    private void RenderSpatialHashAgents()
    {
        if (simAgents == null || simAgents.Length == 0 || !TryResolveInstancedRenderingAssets())
        {
            return;
        }

        for (int partIndex = 0; partIndex < instancedRenderParts.Count; partIndex++)
        {
            InstancedRenderPart renderPart = instancedRenderParts[partIndex];
            int batchCount = 0;

            for (int i = 0; i < simAgents.Length; i++)
            {
                Vector3 forward = simAgents[i].velocity.WithY(0f);
                Quaternion rotation = forward.sqrMagnitude > 0.0001f
                    ? Quaternion.LookRotation(forward.normalized, Vector3.up)
                    : Quaternion.identity;

                Matrix4x4 agentMatrix = Matrix4x4.TRS(
                    simAgents[i].position,
                    rotation,
                    Vector3.one * instancedAgentScale);

                instanceMatrices[batchCount] = agentMatrix * renderPart.localMatrix;
                batchCount++;

                if (batchCount == MaxInstancesPerDrawCall)
                {
                    DrawInstancedBatch(renderPart, batchCount);
                    batchCount = 0;
                }
            }

            if (batchCount > 0)
            {
                DrawInstancedBatch(renderPart, batchCount);
            }
        }
    }

    private void RenderEcsAgents()
    {
        if (!TryGetEntityManager(out EntityManager entityManager) || !TryResolveInstancedRenderingAssets())
        {
            return;
        }

        EnsureEcsQueries(entityManager);

        int agentCount = ecsAgentQuery.CalculateEntityCount();
        if (agentCount == 0)
        {
            return;
        }

        NativeArray<Translation> translations = ecsAgentQuery.ToComponentDataArray<Translation>(Allocator.TempJob);
        NativeArray<Rotation> rotations = ecsAgentQuery.ToComponentDataArray<Rotation>(Allocator.TempJob);

        try
        {
            for (int partIndex = 0; partIndex < instancedRenderParts.Count; partIndex++)
            {
                InstancedRenderPart renderPart = instancedRenderParts[partIndex];
                int batchCount = 0;

                for (int i = 0; i < agentCount; i++)
                {
                    Matrix4x4 agentMatrix = Matrix4x4.TRS(
                        ToVector3(translations[i].Value),
                        ToQuaternion(rotations[i].Value),
                        Vector3.one * instancedAgentScale);

                    instanceMatrices[batchCount] = agentMatrix * renderPart.localMatrix;
                    batchCount++;

                    if (batchCount == MaxInstancesPerDrawCall)
                    {
                        DrawInstancedBatch(renderPart, batchCount);
                        batchCount = 0;
                    }
                }

                if (batchCount > 0)
                {
                    DrawInstancedBatch(renderPart, batchCount);
                }
            }
        }
        finally
        {
            rotations.Dispose();
            translations.Dispose();
        }
    }

    private void DrawInstancedBatch(InstancedRenderPart renderPart, int batchCount)
    {
        int subMeshCount = Mathf.Min(renderPart.mesh.subMeshCount, renderPart.materials.Length);

        for (int subMeshIndex = 0; subMeshIndex < subMeshCount; subMeshIndex++)
        {
            Material drawMaterial = GetRuntimeInstancedMaterial(renderPart.materials[subMeshIndex]);

            if (drawMaterial == null)
            {
                continue;
            }

            Graphics.DrawMeshInstanced(
                renderPart.mesh,
                subMeshIndex,
                drawMaterial,
                instanceMatrices,
                batchCount,
                null,
                renderPart.shadowCastingMode,
                renderPart.receiveShadows,
                renderPart.layer);
        }
    }

    private Material GetRuntimeInstancedMaterial(Material sourceMaterial)
    {
        if (sourceMaterial == null)
        {
            return null;
        }

        if (runtimeInstancedMaterials.TryGetValue(sourceMaterial, out Material runtimeMaterial))
        {
            return runtimeMaterial;
        }

        runtimeMaterial = new Material(sourceMaterial)
        {
            enableInstancing = true,
            hideFlags = HideFlags.DontSave,
            name = $"{sourceMaterial.name} (Instanced Runtime)"
        };

        runtimeInstancedMaterials.Add(sourceMaterial, runtimeMaterial);
        return runtimeMaterial;
    }

    private void ReleaseRuntimeInstancedMaterials()
    {
        foreach (Material material in runtimeInstancedMaterials.Values)
        {
            if (material == null)
            {
                continue;
            }

            if (Application.isPlaying)
            {
                Destroy(material);
            }
            else
            {
                DestroyImmediate(material);
            }
        }

        runtimeInstancedMaterials.Clear();
    }

    private bool TryGetEntityManager(out EntityManager entityManager)
    {
        World world = World.DefaultGameObjectInjectionWorld;

        if (world == null)
        {
            entityManager = default;
            return false;
        }

        entityManager = world.EntityManager;
        return true;
    }

    private void EnsureEcsQueries(EntityManager entityManager)
    {
        if (!ecsQueriesInitialized)
        {
            ecsAgentQuery = entityManager.CreateEntityQuery(
                ComponentType.ReadOnly<CrowdEcsAgent>(),
                ComponentType.ReadOnly<Translation>(),
                ComponentType.ReadOnly<Rotation>());

            ecsSettingsQuery = entityManager.CreateEntityQuery(ComponentType.ReadOnly<CrowdEcsSettings>());
            ecsQueriesInitialized = true;
        }
    }

    private int GetEcsAgentCount()
    {
        RefreshEcsMetricCache(false);
        return cachedEcsAgentCount;
    }

    private int GetEcsStuckAgentCount()
    {
        RefreshEcsMetricCache(false);
        return cachedEcsStuckCount;
    }

    private int GetEcsCompletedTaskCount()
    {
        RefreshEcsMetricCache(false);
        return cachedEcsCompletedTasks;
    }

    private void RefreshEcsMetricCache(bool force)
    {
        if (!TryGetEntityManager(out EntityManager entityManager))
        {
            cachedEcsAgentCount = 0;
            cachedEcsStuckCount = 0;
            cachedEcsCompletedTasks = 0;
            return;
        }

        if (!force && Time.unscaledTime < nextEcsMetricSampleTime)
        {
            return;
        }

        EnsureEcsQueries(entityManager);

        int stuckCount = 0;
        int completedTasks = 0;
        int agentCount = ecsAgentQuery.CalculateEntityCount();

        if (agentCount > 0)
        {
            NativeArray<CrowdEcsAgent> ecsAgents = ecsAgentQuery.ToComponentDataArray<CrowdEcsAgent>(Allocator.TempJob);

            try
            {
                for (int i = 0; i < ecsAgents.Length; i++)
                {
                    stuckCount += ecsAgents[i].IsStuck;
                    completedTasks += ecsAgents[i].CompletedTasks;
                }
            }
            finally
            {
                ecsAgents.Dispose();
            }
        }

        cachedEcsAgentCount = agentCount;
        cachedEcsStuckCount = stuckCount;
        cachedEcsCompletedTasks = completedTasks;
        nextEcsMetricSampleTime = Time.unscaledTime + 0.25f;
    }

    private Vector2Int GetSpatialCell(Vector3 position)
    {
        return new Vector2Int(
            Mathf.FloorToInt(position.x / spatialCellSize),
            Mathf.FloorToInt(position.z / spatialCellSize));
    }

    private Vector3 ClampToSpawnArea(Vector3 position)
    {
        Vector3 center = transform.position;
        float halfWidth = spawnAreaSize.x * 0.5f;
        float halfDepth = spawnAreaSize.y * 0.5f;

        position.x = Mathf.Clamp(position.x, center.x - halfWidth, center.x + halfWidth);
        position.z = Mathf.Clamp(position.z, center.z - halfDepth, center.z + halfDepth);
        return position;
    }

    private Vector2 GetBoundaryDistance(Vector3 position)
    {
        Vector3 center = transform.position;
        float halfWidth = spawnAreaSize.x * 0.5f;
        float halfDepth = spawnAreaSize.y * 0.5f;
        float distanceToXEdge = halfWidth - Mathf.Abs(position.x - center.x);
        float distanceToZEdge = halfDepth - Mathf.Abs(position.z - center.z);

        return new Vector2(distanceToXEdge, distanceToZEdge);
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

    private Vector3 GetRandomSimulationPoint()
    {
        Vector3 areaCenter = transform.position;
        float x = Random.Range(-spawnAreaSize.x * 0.5f, spawnAreaSize.x * 0.5f);
        float z = Random.Range(-spawnAreaSize.y * 0.5f, spawnAreaSize.y * 0.5f);

        return areaCenter + new Vector3(x, 0f, z);
    }

    private static Unity.Mathematics.float3 ToFloat3(Vector3 value)
    {
        return new Unity.Mathematics.float3(value.x, value.y, value.z);
    }

    private static Vector3 ToVector3(Unity.Mathematics.float3 value)
    {
        return new Vector3(value.x, value.y, value.z);
    }

    private static Quaternion ToQuaternion(Unity.Mathematics.quaternion value)
    {
        return new Quaternion(value.value.x, value.value.y, value.value.z, value.value.w);
    }

    private int GetStuckAgentCount()
    {
        if (IsSpatialAlgorithm(simulationAlgorithm))
        {
            int instancedStuckCount = 0;

            if (simAgents == null)
            {
                return instancedStuckCount;
            }

            for (int i = 0; i < simAgents.Length; i++)
            {
                if (simAgents[i].isStuck)
                {
                    instancedStuckCount++;
                }
            }

            return instancedStuckCount;
        }

        if (IsEcsAlgorithm(simulationAlgorithm))
        {
            return GetEcsStuckAgentCount();
        }

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
        if (IsSpatialAlgorithm(simulationAlgorithm))
        {
            int instancedCompletedTasks = 0;

            if (simAgents == null)
            {
                return instancedCompletedTasks;
            }

            for (int i = 0; i < simAgents.Length; i++)
            {
                instancedCompletedTasks += simAgents[i].completedTasks;
            }

            return instancedCompletedTasks;
        }

        if (IsEcsAlgorithm(simulationAlgorithm))
        {
            return GetEcsCompletedTaskCount();
        }

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

internal static class CrowdVectorExtensions
{
    public static Vector3 WithY(this Vector3 value, float y)
    {
        value.y = y;
        return value;
    }
}
