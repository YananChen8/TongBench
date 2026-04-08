#pragma once
#include <condition_variable>
#include <list>
#include <mutex>
#include <optional>

namespace tongos {

template <typename T> class Channel {
public:
  Channel() : done(false) {}

  template <typename U> void send(U &&u) {
    std::lock_guard lock_guard(m);
    queue.push_back(std::forward<U>(u));
    if (queue.size() == 1) {
      cv.notify_one();
    }
  }

  void close() {
    std::lock_guard lock_guard(m);
    done = true;
    cv.notify_one();
  }

  std::optional<T> try_receive() {
    std::unique_lock unique_lock(m);
    if (queue.size() == 0) {
      return {};
    }
    std::optional<T> value = std::move(queue.front());
    queue.pop_front();
    return value;
  }

  std::optional<T> receive() {
    std::unique_lock unique_lock(m);
    while (queue.size() == 0 && !done) {
      cv.wait(unique_lock);
    }
    if (queue.size() == 0 && done) {
      return {};
    }
    std::optional<T> value = std::move(queue.front());
    queue.pop_front();
    return value;
  }

private:
  std::list<T> queue;
  bool done;
  std::mutex m;
  std::condition_variable cv;
};
} // namespace tongos
