use std::sync::Arc;
use tokio::sync::watch;

/// Creates a linked (`CancellationHandle`, `CancellationToken`) pair.
///
/// The handle is given to the *owner* of a task; the token is passed into the
/// task itself.  Dropping the handle does **not** cancel — call
/// [`CancellationHandle::cancel`] explicitly.
pub fn cancellation_pair() -> (CancellationHandle, CancellationToken) {
    let (tx, rx) = watch::channel(false);
    (
        CancellationHandle {
            sender: Arc::new(tx),
        },
        CancellationToken { receiver: rx },
    )
}
/// Allows the owner of a task to request cancellation.
///
/// Clone-able so that multiple parties can independently cancel the same task.
#[derive(Clone)]
pub struct CancellationHandle {
    sender: Arc<watch::Sender<bool>>,
}

impl CancellationHandle {
    /// Signal cancellation to all associated [`CancellationToken`]s.
    pub fn cancel(&self) {
        // Ignore errors: all tokens have been dropped, nothing to signal.
        let _ = self.sender.send(true);
    }

    /// Returns `true` if cancellation has already been requested.
    pub fn is_cancelled(&self) -> bool {
        *self.sender.borrow()
    }
}
/// Passed into a running task so it can observe cancellation requests.
///
/// Clone-able: each clone independently tracks whether it has already seen the
/// cancellation signal (via the `watch` channel's mark-seen semantics).
#[derive(Clone)]
pub struct CancellationToken {
    receiver: watch::Receiver<bool>,
}

impl CancellationToken {
    /// Returns `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        *self.receiver.borrow()
    }

    /// Await until cancellation is *actually* requested, never resolving if it
    /// becomes impossible.
    ///
    /// Differs from [`Self::cancelled`] only in the dropped-handle case:
    /// `cancelled()` resolves (so a task awaiting it alone cannot hang forever),
    /// whereas this parks forever, because nobody can cancel any more.
    ///
    /// That distinction matters when racing cancellation against real work, e.g.
    /// `tokio::select!` over this and a stream's `next()`. Every current caller
    /// builds its context with `let (_handle, ctx) = ...`, dropping the handle
    /// immediately — so a race against `cancelled()` would fire instantly and
    /// abort the very work it was meant to guard. Use this in a `select!`; use
    /// `cancelled()` when it is the only thing being awaited.
    pub async fn cancel_requested(&self) {
        if self.is_cancelled() {
            return;
        }
        let mut rx = self.receiver.clone();
        loop {
            if rx.changed().await.is_err() {
                // Sender dropped: cancellation can never arrive. Park rather
                // than report cancellation, so a `select!` arm built on this
                // never wins spuriously.
                std::future::pending::<()>().await;
            }
            if *rx.borrow() {
                return;
            }
        }
    }

    /// Await until cancellation is requested.
    ///
    /// Returns immediately if already cancelled.  Also returns if the
    /// [`CancellationHandle`] is dropped without cancelling (treat as
    /// cancelled to avoid hanging forever). When racing this against real work,
    /// prefer [`Self::cancel_requested`] — see its docs for why.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let mut rx = self.receiver.clone();
        loop {
            match rx.changed().await {
                Ok(_) => {
                    if *rx.borrow() {
                        return;
                    }
                }
                // Sender dropped — treat as cancelled so tasks don't hang.
                Err(_) => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_cancellation_signal_sync() {
        let (handle, token) = cancellation_pair();

        assert!(
            !token.is_cancelled(),
            "token should not be cancelled initially"
        );

        handle.cancel();

        assert!(
            token.is_cancelled(),
            "token should be cancelled after handle.cancel()"
        );
    }

    #[tokio::test]
    async fn test_cancellation_signal_async() {
        let (handle, token) = cancellation_pair();

        assert!(!token.is_cancelled());

        handle.cancel();

        assert!(token.is_cancelled());

        // `cancelled().await` should return immediately since cancel was already called.
        let result = tokio::time::timeout(Duration::from_millis(100), token.cancelled()).await;

        assert!(
            result.is_ok(),
            "token.cancelled().await should complete immediately after cancel, not time out"
        );
    }

    /// The two waiters differ precisely on a dropped handle, and the difference
    /// is load-bearing: `process_stream` races `cancel_requested()` against a
    /// stream's `next()`, and every caller builds its context with
    /// `let (_handle, ctx) = ..`. If that arm resolved on a dropped handle it
    /// would win immediately and abort every streaming pipeline.
    #[tokio::test]
    async fn dropped_handle_resolves_cancelled_but_never_cancel_requested() {
        let (handle, token) = cancellation_pair();
        drop(handle);

        assert!(
            !token.is_cancelled(),
            "dropping the handle is not a cancellation request"
        );

        // `cancelled()` deliberately resolves, so a lone awaiter cannot hang.
        assert!(
            tokio::time::timeout(Duration::from_millis(100), token.cancelled())
                .await
                .is_ok(),
            "cancelled() should treat a dropped handle as cancelled"
        );

        // `cancel_requested()` must park: nobody can cancel any more.
        assert!(
            tokio::time::timeout(Duration::from_millis(100), token.cancel_requested())
                .await
                .is_err(),
            "cancel_requested() must not resolve on a dropped handle — a select! \
             arm built on it would abort the work it is guarding"
        );
    }

    #[tokio::test]
    async fn cancel_requested_resolves_on_a_real_cancel() {
        let (handle, token) = cancellation_pair();

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            handle.cancel();
        });

        assert!(
            tokio::time::timeout(Duration::from_secs(5), token.cancel_requested())
                .await
                .is_ok(),
            "cancel_requested() must resolve once cancel() is called"
        );
    }
}
