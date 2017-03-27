#ifndef _FV_BQUEUE_H_
#define _FV_BQUEUE_H_

#include <queue>
#include <string>
#include <memory>
#include <condition_variable>
#include <pthread.h>

namespace FEATURE_VERIFYNS {

struct FV_Msg {
public:
    FV_Msg()
        : status_(0)
        , report_(-1)
    {}

    void ChangeStatus(int code, int report = -1){
        std::unique_lock<std::mutex> lock(mutex_);
        status_ = code;
        report_ = report;
        lock.unlock();
        condition_.notify_all();
    }

    mutable std::mutex mutex_;
    std::condition_variable condition_;

    /**
     * -1: thread quit
     *  0: thread idle
     *  1: thread running
     */
    int status_;

    /**
     * -1: no report
     *  0: pre job failed
     *  1: pre job successful
     */
    int report_;
};

template<typename T>
class FV_BQueue {
 public:
  explicit FV_BQueue()
        : sync_(new sync()) {
    }

  void push(const T& t) {
      std::unique_lock<std::mutex> lock(sync_->mutex_);
      queue_.push(t);
      lock.unlock();
      sync_->condition_.notify_all();
    }
  T pop(const std::string& log_on_wait = "") {
      std::unique_lock<std::mutex> lock(sync_->mutex_);
      while (queue_.empty())
        sync_->condition_.wait(lock);      

      T t = queue_.front();
      queue_.pop();
      sync_->condition_.notify_all();
      return t;
    }
  const size_t size() const {
      std::lock_guard<std::mutex> lock(sync_->mutex_);
      return queue_.size();
    }

 protected:
  class sync {
   public:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
  };

  std::queue<T> queue_;
  std::shared_ptr<sync> sync_;

};

} /// end namespace
#endif

