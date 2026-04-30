using UnityEngine;
using UnityEngine.AI;

[RequireComponent(typeof(NavMeshAgent))]
public class CrowdAgent : MonoBehaviour
{
    [SerializeField, Min(0.01f)] private float targetReachedDistance = 1.25f;
    [SerializeField, Min(0f)] private float stuckSpeedThreshold = 0.1f;
    [SerializeField, Min(0f)] private float stuckTimeThreshold = 2f;

    private CrowdExperimentManager experimentManager;
    private NavMeshAgent navMeshAgent;
    private float lowSpeedTimer;
    private bool hasTarget;

    public bool IsStuck { get; private set; }
    public float CurrentSpeed { get; private set; }
    public bool HasReachedTarget { get; private set; }
    public int CompletedTasks { get; private set; }

    private void Awake()
    {
        navMeshAgent = GetComponent<NavMeshAgent>();
    }

    private void Update()
    {
        CurrentSpeed = navMeshAgent.velocity.magnitude;
        HasReachedTarget = CheckReachedTarget();

        if (HasReachedTarget)
        {
            CompletedTasks++;
            hasTarget = false;
            AssignNewTarget();
        }

        UpdateStuckState();
    }

    public void Initialize(CrowdExperimentManager manager)
    {
        experimentManager = manager;
        lowSpeedTimer = 0f;
        IsStuck = false;
        CurrentSpeed = 0f;
        HasReachedTarget = false;
        CompletedTasks = 0;
        hasTarget = false;

        navMeshAgent.isStopped = false;
    }

    public void SetTarget(Vector3 target)
    {
        navMeshAgent.SetDestination(target);
        hasTarget = true;
        HasReachedTarget = false;
    }

    private void AssignNewTarget()
    {
        if (experimentManager != null && experimentManager.TryGetRandomDestination(out Vector3 target))
        {
            SetTarget(target);
        }
    }

    private bool CheckReachedTarget()
    {
        if (navMeshAgent.pathPending)
        {
            return false;
        }

        if (!hasTarget || !navMeshAgent.hasPath)
        {
            return false;
        }

        return navMeshAgent.remainingDistance <= targetReachedDistance;
    }

    private void UpdateStuckState()
    {
        bool hasActivePath = navMeshAgent.hasPath && !navMeshAgent.pathPending && !HasReachedTarget;
        bool movingTooSlowly = CurrentSpeed < stuckSpeedThreshold;

        if (hasActivePath && movingTooSlowly)
        {
            lowSpeedTimer += Time.deltaTime;
            IsStuck = lowSpeedTimer > stuckTimeThreshold;
            return;
        }

        lowSpeedTimer = 0f;
        IsStuck = false;
    }
}
