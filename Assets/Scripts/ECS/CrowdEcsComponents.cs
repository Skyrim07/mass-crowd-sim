using Unity.Entities;
using Unity.Mathematics;

public struct CrowdEcsAgent : IComponentData
{
    public float3 Velocity;
    public float3 Target;
    public float LowSpeedTimer;
    public int IsStuck;
    public int CompletedTasks;
}

public struct CrowdEcsRandom : IComponentData
{
    public Random Value;
}

public struct CrowdEcsSettings : IComponentData
{
    public float3 Center;
    public float2 SpawnAreaSize;
    public float AgentSpeed;
    public float TurnResponsiveness;
    public float SpatialCellSize;
    public float NeighborRadius;
    public float SeparationStrength;
    public float TargetReachedDistance;
    public float StuckSpeedThreshold;
    public float StuckTimeThreshold;
}
