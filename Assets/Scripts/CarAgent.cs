using UnityEngine;
using UnityEngine.InputSystem;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

// Агент, который учится парковать машину.
// Наследуется от Agent (класс из ML-Agents), а не от обычного MonoBehaviour.
public class CarAgent : Agent
{
    [Header("Ссылки")]
    public Transform parkingSpot;          // сюда перетащить объект ParkingSpot

    [Header("Параметры машины")]
    public float maxSpeed = 8f;            // максимальная скорость вперёд (м/с)
    public float acceleration = 10f;       // как быстро разгоняется
    public float steerSpeed = 100f;        // скорость поворота (градусов в секунду)
    public float brakePower = 15f;         // сила тормоза

    [Header("Условия успешной парковки")]
    public float parkDistance = 1.0f;      // насколько близко к центру места
    public float parkAngle = 15f;          // допустимый перекос в градусах
    public float parkSpeed = 0.3f;         // почти полная остановка

    Rigidbody rb;
    float currentSpeed;                    // текущая скорость (минус = задний ход)
    float previousDistance;                // расстояние до места на прошлом шаге

    // Вызывается один раз при старте
    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    // Вызывается в начале каждой попытки (эпизода)
    public override void OnEpisodeBegin()
    {
        // Случайная точка старта в нижней половине площадки
        float x = Random.Range(-10f, 10f);
        float z = Random.Range(-12f, -2f);
        transform.localPosition = new Vector3(x, 0.5f, z);

        // Случайный поворот машины
        transform.localRotation = Quaternion.Euler(0f, Random.Range(-60f, 60f), 0f);

        // Обнуляем скорость
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        currentSpeed = 0f;

        previousDistance = DistanceToSpot();
    }

    // Что агент "знает" о мире (5 чисел) + лучи RayPerceptionSensor добавляются отдельно
    public override void CollectObservations(VectorSensor sensor)
    {
        // Направление на парковку с точки зрения машины (x = влево/вправо, z = вперёд/назад)
        Vector3 toSpot = transform.InverseTransformDirection(parkingSpot.position - transform.position);
        sensor.AddObservation(toSpot.x / 30f);
        sensor.AddObservation(toSpot.z / 30f);

        // Насколько машина повёрнута так же, как парковочное место
        sensor.AddObservation(Vector3.Dot(transform.forward, parkingSpot.forward));
        sensor.AddObservation(Vector3.Dot(transform.right, parkingSpot.forward));

        // Текущая скорость
        sensor.AddObservation(currentSpeed / maxSpeed);
    }

    // Здесь агент получает свои действия и мы двигаем машину
    public override void OnActionReceived(ActionBuffers actions)
    {
        float steer = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);              // руль
        float throttle = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);           // газ / задний ход
        // Тормоз включается только когда третье действие > 0.5 (сила 0..1).
        // Иначе при случайных действиях в начале обучения машина почти всегда тормозит и не едет.
        float brakeInput = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float brake = brakeInput > 0.5f ? (brakeInput - 0.5f) * 2f : 0f;

        float dt = Time.fixedDeltaTime;

        // Разгон и торможение
        currentSpeed += throttle * acceleration * dt;
        currentSpeed = Mathf.MoveTowards(currentSpeed, 0f, brake * brakePower * dt);
        currentSpeed = Mathf.MoveTowards(currentSpeed, 0f, 1f * dt);  // лёгкое трение
        currentSpeed = Mathf.Clamp(currentSpeed, -maxSpeed / 2f, maxSpeed);

        // Поворот: чем быстрее едем, тем сильнее поворачиваем (на месте машина не крутится)
        float turn = steer * steerSpeed * dt * (currentSpeed / maxSpeed);
        rb.MoveRotation(rb.rotation * Quaternion.Euler(0f, turn, 0f));

        // Движение вперёд/назад, гравитацию сохраняем
        rb.linearVelocity = transform.forward * currentSpeed + Vector3.up * rb.linearVelocity.y;

        // ----- Награды -----

        // Маленький штраф за каждый шаг, чтобы агент не тянул время
        if (MaxStep > 0) AddReward(-1f / MaxStep);

        // Награда за приближение к месту (и штраф за удаление)
        float distance = DistanceToSpot();
        AddReward((previousDistance - distance) * 0.1f);
        previousDistance = distance;

        // Проверка успешной парковки
        float angle = Vector3.Angle(transform.forward, parkingSpot.forward);
        bool aligned = angle < parkAngle || angle > 180f - parkAngle;  // можно заехать и передом, и задом

        if (distance < parkDistance && aligned && Mathf.Abs(currentSpeed) < parkSpeed)
        {
            AddReward(2f);  // большая награда за успешную парковку
            EndEpisode();
        }
    }

    // Столкновение со стеной или другой машиной = провал
    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Wall") || collision.gameObject.CompareTag("Obstacle"))
        {
            AddReward(-1f);
            EndEpisode();
        }
    }

    // Ручное управление для проверки: W/S газ, A/D руль, пробел тормоз
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        var kb = Keyboard.current;
        if (kb == null) return;

        a[0] = (kb.dKey.isPressed ? 1f : 0f) - (kb.aKey.isPressed ? 1f : 0f);
        a[1] = (kb.wKey.isPressed ? 1f : 0f) - (kb.sKey.isPressed ? 1f : 0f);
        a[2] = kb.spaceKey.isPressed ? 1f : -1f;
    }

    // Расстояние до центра парковки по земле (без учёта высоты)
    float DistanceToSpot()
    {
        Vector3 a = transform.localPosition;
        Vector3 b = parkingSpot.localPosition;
        a.y = 0f;
        b.y = 0f;
        return Vector3.Distance(a, b);
    }
}
