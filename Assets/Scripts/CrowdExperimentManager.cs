using System.Collections;
using System.Collections.Generic;
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
        DotsEcsGpuInstanced
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

    private const int MaxInstancesPerDrawCall = 1023;

    [Header("Algorithm")]
    [SerializeField] private SimulationAlgorithm simulationAlgorithm = SimulationAlgorithm.BaselineNavMesh;

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

    [Header("Spatial Hash + GPU Instancing")]
    [SerializeField] private Mesh instancedAgentMesh;
    [SerializeField] private Material instancedAgentMaterial;
    [SerializeField, Min(0.01f)] private float instancedAgentScale = 1f;
    [SerializeField, Min(0.01f)] private float instancedAgentSpeed = 3.5f;
    [SerializeField, Min(0.01f)] private float instancedTurnResponsiveness = 8f;
    [SerializeField, Min(0.1f)] private float spatialCellSize = 2.5f;
    [SerializeField, Min(0.1f)] private float neighborRadius = 1.5f;
    [SerializeField, Min(0f)] private float separationStrength = 2.5f;
    [SerializeField, Min(0.01f)] private float instancedTargetReachedDistance = 1.25f;
    [SerializeField, Min(0f)] private float instancedStuckSpeedThreshold = 0.1f;
    [SerializeField, Min(0f)] private float instancedStuckTimeThreshold = 2f;

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
    private float smoothedDeltaTime;
    private bool metricsRunActive;

    public IReadOnlyList<CrowdAgent> Agents => agents;
    public int ActiveAgentCount => simulationAlgorithm == SimulationAlgorithm.BaselineNavMesh
        ? agents.Count
        : simulationAlgorithm == SimulationAlgorithm.SpatialHashGpuInstanced
            ? simAgents?.Length ?? 0
            : GetEcsAgentCount();

    private void Update()
    {
        smoothedDeltaTime += (Time.unscaledDeltaTime - smoothedDeltaTime) * 0.1f;

        if (simulationAlgorithm == SimulationAlgorithm.SpatialHashGpuInstanced)
        {
            UpdateSpatialHashSimulation(Time.deltaTime);
        }
    }

    private void LateUpdate()
    {
        if (simulationAlgorithm == SimulationAlgorithm.SpatialHashGpuInstanced)
        {
            RenderSpatialHashAgents();
        }

        if (simulationAlgorithm == SimulationAlgorithm.DotsEcsGpuInstanced)
        {
            RenderEcsAgents();
        }
    }

    private void Start()
    {
        if (spawnOnStart)
        {
            ResetExperiment();
            BeginMetricsRun();
        }
    }

    private void OnDisable()
    {
        EndMetricsRun();
        ReleaseRuntimeInstancedMaterials();
        ClearEcsAgents();
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
                metricsLogger.BeginRun(simulationAlgorithm.ToString(), testAgentCount, GetTotalCompletedTasks);
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

        if (GUILayout.Button("DOTS ECS"))
        {
            SwitchAlgorithm(SimulationAlgorithm.DotsEcsGpuInstanced);
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

        if (simulationAlgorithm == SimulationAlgorithm.DotsEcsGpuInstanced)
        {
            InitializeEcsAgents(count);
            return;
        }

        if (simulationAlgorithm == SimulationAlgorithm.SpatialHashGpuInstanced)
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

    private void BeginMetricsRun()
    {
        if (metricsLogger == null)
        {
            return;
        }

        metricsLogger.BeginRun(simulationAlgorithm.ToString(), ActiveAgentCount, GetTotalCompletedTasks);
        metricsRunActive = true;
    }

    private void EndMetricsRun()
    {
        if (!metricsRunActive || metricsLogger == null)
        {
            return;
        }

        metricsLogger.EndRun();
        metricsRunActive = false;
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

    private void InitializeEcsAgents(int count)
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
            StuckTimeThreshold = instancedStuckTimeThreshold
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
            if (!TryGetRandomNavMeshPoint(out Vector3 spawnPosition))
            {
                Debug.LogWarning($"Could not find a valid spawn point for DOTS ECS agent {i}.");
                spawnPosition = transform.position;
            }

            if (!TryGetRandomDestination(out Vector3 destination))
            {
                destination = transform.position;
            }

            uint randomStateSeed = (uint)Mathf.Max(1, randomSeed + i + 1);
            Unity.Mathematics.Random randomState = new Unity.Mathematics.Random(randomStateSeed);
            Vector3 initialVelocity = Random.insideUnitSphere.WithY(0f).normalized * instancedAgentSpeed;

            Entity agentEntity = entityManager.CreateEntity(agentArchetype);
            entityManager.SetComponentData(agentEntity, new CrowdEcsAgent
            {
                Velocity = ToFloat3(initialVelocity),
                Target = ToFloat3(destination),
                LowSpeedTimer = 0f,
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
            if (!TryGetRandomNavMeshPoint(out Vector3 spawnPosition))
            {
                Debug.LogWarning($"Could not find a valid spawn point for instanced agent {i}.");
                spawnPosition = transform.position;
            }

            if (!TryGetRandomDestination(out Vector3 destination))
            {
                destination = transform.position;
            }

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

            desiredVelocity += CalculateSeparation(i, agent.position, neighborRadiusSqr, cellSearchRadius) * separationStrength;

            if (desiredVelocity.sqrMagnitude > instancedAgentSpeed * instancedAgentSpeed)
            {
                desiredVelocity = desiredVelocity.normalized * instancedAgentSpeed;
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

    private Vector3 CalculateSeparation(int agentIndex, Vector3 position, float neighborRadiusSqr, int cellSearchRadius)
    {
        Vector2Int centerCell = GetSpatialCell(position);
        Vector3 separation = Vector3.zero;

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

                    if (distanceSqr <= 0.0001f || distanceSqr > neighborRadiusSqr)
                    {
                        continue;
                    }

                    float distance = Mathf.Sqrt(distanceSqr);
                    separation += offset / distance * (1f - distance / neighborRadius);
                }
            }
        }

        return separation;
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
        if (simulationAlgorithm == SimulationAlgorithm.SpatialHashGpuInstanced)
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

        if (simulationAlgorithm == SimulationAlgorithm.DotsEcsGpuInstanced)
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
        if (simulationAlgorithm == SimulationAlgorithm.SpatialHashGpuInstanced)
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

        if (simulationAlgorithm == SimulationAlgorithm.DotsEcsGpuInstanced)
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
