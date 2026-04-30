using UnityEngine;

[CreateAssetMenu(fileName = "CrowdExperimentConfig", menuName = "Crowd Simulation/Experiment Config")]
public class ExperimentConfig : ScriptableObject
{
    public enum SimulationVariant
    {
        NaiveEveryAgentEveryFrame,
        CentralizedScheduling,
        SpatialPartitioning,
        BehaviorLOD
    }

    [Header("Experiment")]
    public SimulationVariant variant = SimulationVariant.NaiveEveryAgentEveryFrame;
    public bool autoStart = true;
    public int randomSeed = 12345;
    public float durationSeconds = 0f;

    [Header("Agents")]
    public CrowdAgent agentPrefab;
    [Min(1)] public int agentCount = 100;
    public Vector2 speedRange = new Vector2(2.5f, 4.0f);
    public float destinationReachedDistance = 1.25f;

    [Header("Scene Sampling")]
    public float spawnRadius = 25f;
    public float destinationRadius = 25f;
    public int navMeshSampleAttempts = 12;
    public float navMeshSampleMaxDistance = 4f;

    [Header("Metrics")]
    public bool enableMetrics = true;
    public float metricsSampleInterval = 1f;

    [Header("Future Optimization Hooks")]
    [Min(1)] public int centralizedUpdatesPerFrame = 256;
    public float spatialCellSize = 5f;
    public float lodNearDistance = 15f;
    public float lodFarDistance = 45f;
}
