#pragma once

#include "server-task.h"

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_set>

#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue {
private:
    int id = 0;
    bool running  = false;
    // latched by terminate(): start_loop checks it under the queue lock and
    // returns immediately on a terminated queue instead of resurrecting it
    bool terminated = false;
    bool sleeping = false;
    bool req_stop_sleeping = false;
    int64_t time_last_task = 0;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;
    // tasks declined while yielding, put back in queue_tasks once the yield is done
    // note: kept as a member so that cleanup_pending_task() can also reach them
    std::deque<server_task> queue_tasks_unhandled;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // used by yield_to_queue, all fields are guarded by mutex_tasks
    struct worker_t {
        std::thread             thread;
        std::condition_variable cv;        // the worker sleeps on this until a yield starts
        std::exception_ptr      exception; // exception thrown while processing tasks, if any
        bool stop     = false;
        bool busy     = false; // set by yield_to_queue(), cleared by the worker once it is done processing tasks
        bool yielding = false; // work() is still running on the start_loop() thread
    };
    worker_t worker;

    // callback functions
    std::function<bool(server_task &&, bool)> callback_new_task;
    std::function<void(void)>                 callback_update_slots;
    std::vector<std::function<void(bool)>>    callback_sleeping_state;

public:
    ~server_queue() { worker_stop(); }

    // Add a new task to the end of the queue
    int post(server_task && task, bool front = false);

    // multi-task version of post()
    int post(std::vector<server_task> && tasks, bool front = false);

    // Add a new task, but defer until one slot is available
    void defer(server_task && task);

    // Get the next id for creating a new task
    int get_new_id();

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
    void pop_deferred_task(int id_slot);

    // if sleeping, request exiting sleep state and wait until it is done
    // returns immediately if not sleeping
    void wait_until_no_sleep();

    bool is_sleeping() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return sleeping;
    }

    // end the start_loop routine
    void terminate();

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     *
     * Sleeping procedure (disabled if idle_sleep_ms < 0):
     * - If there is no task after idle_sleep_ms, enter sleeping state
     *   note: metrics tasks are processed as usual, but do not reset the idle timer
     * - Call callback_sleeping_state(true)
     * - Wait until req_stop_sleeping is set to true
     * - Call callback_sleeping_state(false)
     * - Exit sleeping state
     */
    void start_loop(int64_t idle_sleep_ms = -1);

    // while waiting for work() to finish, run process_new_tasks on the worker thread
    // returns once work() is done (may throw exceptions)
    // must be called from start_loop() thread (ideally inside callback_update_slots)
    // use case: return metrics while encode/decode is running
    // ref: https://github.com/ggml-org/llama.cpp/pull/27041
    //
    // tasks declined by callback_new_task are put back in the queue once this returns
    void yield_to_queue(std::function<void()> && work);

    // for metrics
    size_t queue_tasks_deferred_size() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return queue_tasks_deferred.size();
    }

    //
    // Functions below are not thread-safe, must only be used before start_loop() is called
    //

    // Register function to process a new task
    // the second argument tells whether the queue is currently yielding (see yield_to_queue)
    // only then may the callback return false to decline the task, and it must leave it
    // untouched, so that it can be put back in the queue later
    // note: while yielding, the callback runs on worker thread, not main thread
    void on_new_task(std::function<bool(server_task &&, bool)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Register callback for sleeping state change; multiple callbacks are allowed
    // for example: register order cb0, cb1, cb2
    // entering sleep: queue.sleeping = true --> cb0(true) --> cb1(true) --> cb2(true)
    // leaving sleep: cb2(false) --> cb1(false) --> cb0(false) --> queue.sleeping = false
    // note: caller will hold mutex_tasks while calling the callbacks
    void on_sleeping_state(std::function<void(bool)> callback) {
        callback_sleeping_state.push_back(std::move(callback));
    }

private:
    void cleanup_pending_task(int id_target);

    // process all pending tasks in the queue
    // returns true if the queue is terminated, false if there is no more task to process
    // while yielding, declined tasks are moved to queue_tasks_unhandled
    bool process_new_tasks(bool is_yielding);

    // for worker_t
    void worker_loop();
    void worker_stop();
};

// result queue for scheduler tasks: holds any result handle whose element type
// exposes the task id it answers (broadcast additionally needs clone(), and is
// only compiled when called). server_task_result_ptr is the production handle.
template <typename T>
struct server_result_queue {
    // the queued handle must expose the task id it answers
    static_assert(std::is_same_v<decltype(std::declval<typename T::element_type>().id), int>,
                  "server_result_queue element type must have an int id field");

private:
    bool running = true;

    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::vector<T> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

public:
    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task) {
        RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());

        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.insert(id_task);
    }

    void add_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (const auto & id_task : id_tasks) {
            RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());
            waiting_task_ids.insert(id_task);
        }
    }

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task) {
        RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());

        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.erase(id_task);
        // make sure to clean up all pending results
        queue_results.erase(
            std::remove_if(queue_results.begin(), queue_results.end(), [id_task](const T & res) {
                return res->id == id_task;
            }),
            queue_results.end());
    }

    // remove multiple tasks from waiting list
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (const auto & id_task : id_tasks) {
            RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());
            waiting_task_ids.erase(id_task);
        }
    }

    // This function blocks the thread until there is a response for one of the id_tasks
    // locked scan-and-take shared by recv() and recv_with_timeout(): removes
    // and returns the first queued result for one of the tasks, or nullptr
    // when none is queued. caller holds mutex_results.
    T take_locked(const std::unordered_set<int> & id_tasks) {
        for (size_t i = 0; i < queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                T res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }
        return nullptr;
    }

    T recv(const std::unordered_set<int> & id_tasks) {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_results);
            condition_results.wait(lock, [&]{
                if (!running) {
                    RES_DBG("%s : queue result stop\n", "recv");
                    std::terminate(); // we cannot return here since the caller is HTTP code
                }
                return !queue_results.empty();
            });

            if (T res = take_locked(id_tasks)) {
                return res;
            }
        }

        // should never reach here
    }

    // same as recv(), but have timeout in seconds
    // if timeout is reached, nullptr is returned
    T recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout) {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_results);

            if (T res = take_locked(id_tasks)) {
                return res;
            }

            std::cv_status cr_res = condition_results.wait_for(lock, std::chrono::seconds(timeout));
            if (!running) {
                RES_DBG("%s : queue result stop\n", __func__);
                std::terminate(); // we cannot return here since the caller is HTTP code
            }
            if (cr_res == std::cv_status::timeout) {
                return nullptr;
            }
        }

        // should never reach here
    }

    // single-task version of recv()
    T recv(int id_task) {
        std::unordered_set<int> id_tasks = {id_task};
        return recv(id_tasks);
    }

    // Send a new result to a waiting id_task. a result with no waiter means
    // the waiter went away (a lost add_waiting_task_id or a timed-out reader):
    // loud on purpose, so a dropped response can never vanish silently.
    void send(T && result) {
        RES_DBG("sending result for task id = %d\n", result->id);

        std::unique_lock<std::mutex> lock(mutex_results);
        for (const auto & id_task : waiting_task_ids) {
            if (result->id == id_task) {
                RES_DBG("task id = %d pushed to result queue\n", result->id);

                queue_results.emplace_back(std::move(result));
                condition_results.notify_all();
                return;
            }
        }
        RES_WRN("no waiter for task id = %d, result dropped\n", result->id);
    }

    // broadcast a new result to all waiting tasks
    // (used by router mode)
    void broadcast(T && result) {
        std::unique_lock<std::mutex> lock(mutex_results);
        for (const auto & id_task : waiting_task_ids) {
            RES_DBG("task id = %d pushed to result queue\n", id_task);
            T res_copy(result->clone());
            res_copy->id = id_task; // override id with target task id
            queue_results.emplace_back(std::move(res_copy));
        }
        condition_results.notify_all();
    }

    // terminate the waiting loop
    void terminate() {
        running = false;
        condition_results.notify_all();
    }
};

// polling cadence for scheduler waits through server_response_reader
constexpr int HTTP_POLLING_SECONDS = 1;

// RAII wrapper to make working with server_queue and server_result_queue easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_queue & queue_tasks;
    server_result_queue<server_task_result_ptr> & queue_results;
    size_t received_count = 0;
    bool cancelled = false;
    int polling_interval_seconds;

    // tracking generation state and partial tool calls
    // only used by streaming completions
    std::vector<task_result_state> states;

    // should_stop function will be called each polling_interval_seconds
    server_response_reader(server_queue & queue_tasks, server_result_queue<server_task_result_ptr> & queue_results, int polling_interval_seconds)
        : queue_tasks(queue_tasks), queue_results(queue_results), polling_interval_seconds(polling_interval_seconds) {}
    ~server_response_reader() {
        stop();
    }

    int get_new_id() {
        return queue_tasks.get_new_id();
    }

    // if front = true, the task will be posted to the front of the queue (high priority)
    void post_task(server_task && task, bool front = false);
    void post_tasks(std::vector<server_task> && tasks, bool front = false);
    bool has_next() const;

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop);

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };
    // aggregate multiple results
    batch_response wait_for_all(const std::function<bool()> & should_stop);

    void stop();
};
