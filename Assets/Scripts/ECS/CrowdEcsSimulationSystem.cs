using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;

[UpdateInGroup(typeof(SimulationSystemGroup))]
public partial class CrowdEcsSimulationSystem : SystemBase
{
    private EntityQuery agentQuery;
    private NativeArray<float3> positions;
    private NativeParallelMultiHashMap<long, int> spatialGrid;

    protected override void OnCreate()
    {
        agentQuery = GetEntityQuery(
            ComponentType.ReadWrite<CrowdEcsAgent>(),
            ComponentType.ReadWrite<CrowdEcsRandom>(),
            ComponentType.ReadWrite<Translation>(),
            ComponentType.ReadWrite<Rotation>());

        RequireSingletonForUpdate<CrowdEcsSettings>();
    }

    protected override void OnDestroy()
    {
        Dependency.Complete();

        if (positions.IsCreated)
        {
            positions.Dispose();
        }

        if (spatialGrid.IsCreated)
        {
            spatialGrid.Dispose();
        }
    }

    protected override void OnUpdate()
    {
        int agentCount = agentQuery.CalculateEntityCount();

        if (agentCount == 0)
        {
            return;
        }

        CrowdEcsSettings settings = GetSingleton<CrowdEcsSettings>();
        float deltaTime = Time.DeltaTime;

        Dependency.Complete();
        EnsureNativeCapacity(agentCount);
        spatialGrid.Clear();

        JobHandle copyPositionsHandle = new CopyPositionsJob
        {
            TranslationType = GetComponentTypeHandle<Translation>(true),
            Positions = positions
        }.ScheduleParallel(agentQuery, Dependency);

        JobHandle buildGridHandle = new BuildSpatialGridJob
        {
            Positions = positions,
            SpatialGrid = spatialGrid.AsParallelWriter(),
            SpatialCellSize = settings.SpatialCellSize
        }.Schedule(agentCount, 64, copyPositionsHandle);

        Dependency = new SimulateAgentsJob
        {
            AgentType = GetComponentTypeHandle<CrowdEcsAgent>(false),
            RandomType = GetComponentTypeHandle<CrowdEcsRandom>(false),
            TranslationType = GetComponentTypeHandle<Translation>(false),
            RotationType = GetComponentTypeHandle<Rotation>(false),
            Positions = positions,
            SpatialGrid = spatialGrid,
            Settings = settings,
            DeltaTime = deltaTime
        }.ScheduleParallel(agentQuery, buildGridHandle);
    }

    private void EnsureNativeCapacity(int agentCount)
    {
        if (!positions.IsCreated || positions.Length < agentCount)
        {
            if (positions.IsCreated)
            {
                positions.Dispose();
            }

            positions = new NativeArray<float3>(agentCount, Allocator.Persistent);
        }

        if (!spatialGrid.IsCreated || spatialGrid.Capacity < agentCount)
        {
            if (spatialGrid.IsCreated)
            {
                spatialGrid.Dispose();
            }

            spatialGrid = new NativeParallelMultiHashMap<long, int>(agentCount, Allocator.Persistent);
        }
    }

    [BurstCompile]
    private struct CopyPositionsJob : IJobChunk
    {
        [ReadOnly] public ComponentTypeHandle<Translation> TranslationType;
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float3> Positions;

        public void Execute(ArchetypeChunk chunk, int chunkIndex, int firstEntityIndex)
        {
            NativeArray<Translation> translations = chunk.GetNativeArray(TranslationType);

            for (int i = 0; i < chunk.Count; i++)
            {
                Positions[firstEntityIndex + i] = translations[i].Value;
            }
        }
    }

    [BurstCompile]
    private struct BuildSpatialGridJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Positions;
        public NativeParallelMultiHashMap<long, int>.ParallelWriter SpatialGrid;
        public float SpatialCellSize;

        public void Execute(int index)
        {
            SpatialGrid.Add(GetSpatialCellKey(Positions[index], SpatialCellSize), index);
        }
    }

    [BurstCompile]
    private struct SimulateAgentsJob : IJobChunk
    {
        public ComponentTypeHandle<CrowdEcsAgent> AgentType;
        public ComponentTypeHandle<CrowdEcsRandom> RandomType;
        public ComponentTypeHandle<Translation> TranslationType;
        public ComponentTypeHandle<Rotation> RotationType;

        [ReadOnly] public NativeArray<float3> Positions;
        [ReadOnly] public NativeParallelMultiHashMap<long, int> SpatialGrid;
        public CrowdEcsSettings Settings;
        public float DeltaTime;

        public void Execute(ArchetypeChunk chunk, int chunkIndex, int firstEntityIndex)
        {
            NativeArray<CrowdEcsAgent> agents = chunk.GetNativeArray(AgentType);
            NativeArray<CrowdEcsRandom> randomStates = chunk.GetNativeArray(RandomType);
            NativeArray<Translation> translations = chunk.GetNativeArray(TranslationType);
            NativeArray<Rotation> rotations = chunk.GetNativeArray(RotationType);

            float neighborRadiusSqr = Settings.NeighborRadius * Settings.NeighborRadius;
            float targetReachedDistanceSqr = Settings.TargetReachedDistance * Settings.TargetReachedDistance;
            float steeringBlend = 1f - math.exp(-Settings.TurnResponsiveness * DeltaTime);
            int cellSearchRadius = math.max(1, (int)math.ceil(Settings.NeighborRadius / Settings.SpatialCellSize));

            for (int i = 0; i < chunk.Count; i++)
            {
                int agentIndex = firstEntityIndex + i;
                CrowdEcsAgent agent = agents[i];
                CrowdEcsRandom randomState = randomStates[i];
                float3 position = translations[i].Value;
                float3 toTarget = Flatten(agent.Target - position);

                if (math.lengthsq(toTarget) <= targetReachedDistanceSqr)
                {
                    agent.CompletedTasks++;
                    agent.Target = GetRandomDestination(Settings, ref randomState.Value);
                    toTarget = Flatten(agent.Target - position);
                }

                float3 desiredVelocity = math.lengthsq(toTarget) > 0.0001f
                    ? math.normalizesafe(toTarget) * Settings.AgentSpeed
                    : float3.zero;

                desiredVelocity += CalculateSeparation(agentIndex, position, Positions, Settings, SpatialGrid, neighborRadiusSqr, cellSearchRadius)
                    * Settings.SeparationStrength;

                float maxSpeedSqr = Settings.AgentSpeed * Settings.AgentSpeed;
                if (math.lengthsq(desiredVelocity) > maxSpeedSqr)
                {
                    desiredVelocity = math.normalizesafe(desiredVelocity) * Settings.AgentSpeed;
                }

                float3 previousPosition = position;
                agent.Velocity = math.lerp(agent.Velocity, desiredVelocity, steeringBlend);
                position += agent.Velocity * DeltaTime;
                position = ClampToSpawnArea(position, Settings);

                float currentSpeed = DeltaTime > 0f
                    ? math.length(position - previousPosition) / DeltaTime
                    : 0f;

                bool hasTarget = math.lengthsq(Flatten(agent.Target - position)) > targetReachedDistanceSqr;
                bool movingTooSlowly = currentSpeed < Settings.StuckSpeedThreshold;

                if (hasTarget && movingTooSlowly)
                {
                    agent.LowSpeedTimer += DeltaTime;
                    agent.IsStuck = agent.LowSpeedTimer > Settings.StuckTimeThreshold ? 1 : 0;
                }
                else
                {
                    agent.LowSpeedTimer = 0f;
                    agent.IsStuck = 0;
                }

                translations[i] = new Translation { Value = position };
                rotations[i] = new Rotation
                {
                    Value = math.lengthsq(agent.Velocity) > 0.0001f
                        ? quaternion.LookRotationSafe(math.normalizesafe(Flatten(agent.Velocity)), new float3(0f, 1f, 0f))
                        : quaternion.identity
                };

                agents[i] = agent;
                randomStates[i] = randomState;
            }
        }
    }

    private static float3 CalculateSeparation(
        int agentIndex,
        float3 position,
        NativeArray<float3> positions,
        CrowdEcsSettings settings,
        NativeParallelMultiHashMap<long, int> spatialGrid,
        float neighborRadiusSqr,
        int cellSearchRadius)
    {
        int2 centerCell = GetSpatialCell(position, settings.SpatialCellSize);
        float3 separation = float3.zero;

        for (int xOffset = -cellSearchRadius; xOffset <= cellSearchRadius; xOffset++)
        {
            for (int yOffset = -cellSearchRadius; yOffset <= cellSearchRadius; yOffset++)
            {
                int2 neighborCell = centerCell + new int2(xOffset, yOffset);
                long neighborKey = GetSpatialCellKey(neighborCell);

                if (!spatialGrid.TryGetFirstValue(neighborKey, out int otherIndex, out NativeParallelMultiHashMapIterator<long> iterator))
                {
                    continue;
                }

                do
                {
                    if (otherIndex == agentIndex)
                    {
                        continue;
                    }

                    float3 offset = Flatten(position - positions[otherIndex]);
                    float distanceSqr = math.lengthsq(offset);

                    if (distanceSqr <= 0.0001f || distanceSqr > neighborRadiusSqr)
                    {
                        continue;
                    }

                    float distance = math.sqrt(distanceSqr);
                    separation += offset / distance * (1f - distance / settings.NeighborRadius);
                }
                while (spatialGrid.TryGetNextValue(out otherIndex, ref iterator));
            }
        }

        return separation;
    }

    private static float3 GetRandomDestination(CrowdEcsSettings settings, ref Random random)
    {
        float halfWidth = settings.SpawnAreaSize.x * 0.5f;
        float halfDepth = settings.SpawnAreaSize.y * 0.5f;

        return settings.Center + new float3(
            random.NextFloat(-halfWidth, halfWidth),
            0f,
            random.NextFloat(-halfDepth, halfDepth));
    }

    private static float3 ClampToSpawnArea(float3 position, CrowdEcsSettings settings)
    {
        float halfWidth = settings.SpawnAreaSize.x * 0.5f;
        float halfDepth = settings.SpawnAreaSize.y * 0.5f;

        position.x = math.clamp(position.x, settings.Center.x - halfWidth, settings.Center.x + halfWidth);
        position.z = math.clamp(position.z, settings.Center.z - halfDepth, settings.Center.z + halfDepth);
        return position;
    }

    private static long GetSpatialCellKey(float3 position, float spatialCellSize)
    {
        return GetSpatialCellKey(GetSpatialCell(position, spatialCellSize));
    }

    private static long GetSpatialCellKey(int2 cell)
    {
        unchecked
        {
            return ((long)cell.x << 32) ^ (uint)cell.y;
        }
    }

    private static int2 GetSpatialCell(float3 position, float spatialCellSize)
    {
        return new int2(
            (int)math.floor(position.x / spatialCellSize),
            (int)math.floor(position.z / spatialCellSize));
    }

    private static float3 Flatten(float3 value)
    {
        value.y = 0f;
        return value;
    }
}
