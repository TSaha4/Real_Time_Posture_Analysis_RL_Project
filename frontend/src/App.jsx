import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import useRLData from "./hooks/useRLData";
import "./App.css";

const API = "http://127.0.0.1:8000";
const QUESTION_SECONDS = 180;

function App() {
  const [screen, setScreen] = useState("home");
  const [toast, setToast] = useState(null);
  const [examProgress, setExamProgress] = useState(100);
  const [finalSummary, setFinalSummary] = useState(null);
  const data = useRLData(true);
  const backendOnline = Boolean(data?.connected);
  const calibrationDone = Boolean(data?.calibration_snapshot);
  const streamSrc = useMemo(() => `${API}/video_feed?screen=${screen}`, [screen]);

  const showToast = useCallback((message, type = "info") => {
    setToast({ message, type });
  }, []);

  const resetToHome = useCallback(async () => {
    try {
      await axios.get(`${API}/stop_session`);
    } catch (err) {
      console.debug("stop_session skipped:", err?.message);
    }
    setFinalSummary(null);
    setExamProgress(100);
    setScreen("home");
  }, []);

  useEffect(() => {
    if (!toast?.message) return;
    const timer = setTimeout(() => setToast(null), 3000);
    return () => clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (screen !== "calibration") return;
    axios.get(`${API}/calibrate`).catch((err) => {
      console.debug("calibrate on enter failed:", err?.message);
    });
  }, [screen]);

  const goToExam = async () => {
    if (!calibrationDone) {
      setScreen("calibration");
      showToast("Calibration is compulsory before the interview starts.", "error");
      return;
    }

    try {
      await axios.get(`${API}/start_exam`);
      setFinalSummary(null);
      setScreen("exam");
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not start the exam.", "error");
    }
  };

  const leaveSession = async () => {
    await resetToHome();
  };

  const headerTitle =
    screen === "calibration"
      ? "Calibration Room"
      : screen === "exam"
        ? "Interview Room"
        : "Interview Summary";

  return (
    <div className={screen === "home" ? "welcome-screen" : "container"}>
      {screen === "home" ? (
        <>
          <div className="bg-glow blue" />
          <div className="bg-glow purple" />
          <div className="hero-section">
            <div className="badge-chip">Website Interview Workflow</div>
            <h2 className="glitch-text">Home - Calibration - Exam</h2>
            <p className="hero-subtitle">
              Complete calibration first, then answer 3 HR questions with live posture monitoring, trust score updates, and final scoring.
            </p>

            {toast?.message ? <div className={`toast-banner ${toast.type || "info"} home-toast`}>{toast.message}</div> : null}

            <div className="mode-selector-grid">
              <div className="glass-morphism mode-card" onClick={() => setScreen("calibration")}>
                <div className="icon-wrapper">CAL</div>
                <h3>Calibration Room</h3>
                <p>Create the reference posture baseline that every later frame is compared against.</p>
                <button className="start-btn posture-btn">Start Calibration</button>
              </div>

              <div
                className={`glass-morphism mode-card ${!calibrationDone ? "disabled-card" : ""}`}
                onClick={goToExam}
              >
                <div className="icon-wrapper">EXM</div>
                <h3>Interview Exam</h3>
                <p>Answer 3 timed HR questions while the backend scores posture quality, answer duration, and focus.</p>
                <button className="start-btn proctor-btn" disabled={!backendOnline}>
                  {calibrationDone ? "Start Interview" : "Calibration Required"}
                </button>
              </div>
            </div>

            <div className="home-status">
              <span className={`status-pill ${backendOnline ? "online" : "offline"}`}>
                Backend {backendOnline ? "Online" : "Offline"}
              </span>
              <span className={`status-pill ${calibrationDone ? "online" : "pending"}`}>
                Calibration {calibrationDone ? "Ready" : "Pending"}
              </span>
            </div>
          </div>
        </>
      ) : (
        <>
          <header className="header">
            <div className="header-title">
              <h1>{headerTitle}</h1>
              <span className="status-badge active">Backend: {data?.mode || "loading"}</span>
            </div>
            <div className="exam-controls">
              {screen !== "summary" ? (
                <button
                  className="exam-btn secondary"
                  onClick={() => setScreen(screen === "calibration" ? "exam" : "calibration")}
                >
                  {screen === "calibration" ? "Go to Exam" : "Go to Calibration"}
                </button>
              ) : null}
              <button className="toggle-btn btn-danger" onClick={leaveSession}>
                End Session
              </button>
            </div>
          </header>

          {toast?.message ? <div className={`toast-banner ${toast.type || "info"}`}>{toast.message}</div> : null}

          {screen === "summary" ? (
            <SummaryPanel summary={finalSummary} onReturnHome={resetToHome} />
          ) : (
            <div className="main-content">
              <div className="left-panel">
                <div className="video-frame-container full-height">
                  {screen === "exam" ? (
                    <div className="camera-progress-shell">
                      <div className="camera-progress-fill" style={{ width: `${examProgress}%` }} />
                    </div>
                  ) : null}
                  {screen === "calibration" && data?.calibration_frozen ? (
                    <div className="camera-freeze-overlay">
                      <div className="freeze-spinner" />
                      <p>Locking reference... {data.calibration_freeze_remaining?.toFixed(1)}s</p>
                    </div>
                  ) : null}
                  <img key={screen} src={streamSrc} alt="Live Feed" className="video-element" />
                </div>
              </div>

              <div className="dashboard-sidebar">
                {!backendOnline ? (
                  <div className="card">
                    <h3>Backend status</h3>
                    <p>Backend seems offline or not responding.</p>
                    <p style={{ opacity: 0.85 }}>Start the backend on port 8000, then reload this page.</p>
                  </div>
                ) : null}
                {screen === "calibration" ? (
                  <CalibrationPanel data={data} showToast={showToast} onGoExam={goToExam} />
                ) : (
                  <ExamPanel
                    data={data}
                    showToast={showToast}
                    setExamProgress={setExamProgress}
                    onExamFinished={(summary) => {
                      setFinalSummary(summary);
                      setScreen("summary");
                    }}
                  />
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function CalibrationPanel({ data, showToast, onGoExam }) {
  const ratioPct = Math.round((data?.face_inside_ratio || 0) * 100);
  const ready = Boolean(data?.calibration_ready);
  const isCalibrating = data?.mode === "calibrating" || data?.mode === "calibration_freeze";
  const snapshot = data?.calibration_snapshot;
  const freezeActive = Boolean(data?.calibration_frozen);

  const resetCalibration = async () => {
    try {
      await axios.get(`${API}/calibrate`);
      showToast("Calibration reset. Align your face inside the oval.", "info");
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not reset calibration.", "error");
    }
  };

  const captureReference = async () => {
    try {
      await axios.get(`${API}/capture_reference`);
      showToast("Reference captured successfully.", "success");
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not capture reference.", "error");
    }
  };

  return (
    <>
      <div className="card">
        <h3>Calibration Status</h3>
        {isCalibrating ? (
          <>
            <p>Face inside oval: <strong>{ratioPct}%</strong></p>
            <p>Capture rule: keep most of your face inside the oval.</p>
            <p>Freeze status: <strong>{freezeActive ? `Locking (${data?.calibration_freeze_remaining?.toFixed(1)}s)` : "Idle"}</strong></p>
            <p>Ready: <strong>{ready ? "Yes" : "No"}</strong></p>
            <p>Face Width: {Math.round(data?.face_width || 0)}</p>
            <p>Face Height: {Math.round(data?.face_height || 0)}</p>
            <p>Head Angle: {(data?.head_angle ?? 0).toFixed(2)}</p>
            <p>Eye Direction: {(data?.eye_dir ?? 0).toFixed(2)}</p>
          </>
        ) : (
          <p style={{ color: "#f2c94c" }}>Calibration complete. You can reset it or move on to the interview.</p>
        )}
        <p>Mode: <strong>{data?.mode || "loading"}</strong></p>
      </div>

      {snapshot ? (
        <div className="card">
          <h3>Saved Baseline</h3>
          <p>Face Width: {Math.round(snapshot.face_w || 0)}</p>
          <p>Face Height: {Math.round(snapshot.face_h || 0)}</p>
          <p>Head Angle: {(snapshot.head_angle ?? 0).toFixed(2)}</p>
          <p>Eye Direction: {(snapshot.eye_dir ?? 0).toFixed(2)}</p>
          <p>Eye Ratio: {(snapshot.eye_ratio ?? 0).toFixed(2)}</p>
          <p>Eye Distance: {(snapshot.eye_dist ?? 0).toFixed(2)}</p>
        </div>
      ) : null}

      <div className="card">
        <h3>Controls</h3>
        <div className="exam-controls">
          <button className="exam-btn secondary" onClick={resetCalibration}>Reset</button>
          <button className="exam-btn primary" onClick={captureReference} disabled={!ready || !isCalibrating}>
            Capture Reference
          </button>
          <button className="exam-btn primary" onClick={onGoExam} disabled={!snapshot}>
            Start Interview
          </button>
        </div>
      </div>

      <div className="card">
        <h3>Live Suggestion</h3>
        <div className="advice-box">{data?.suggestion || "Waiting for camera data..."}</div>
      </div>

      <AlgorithmPanel data={data} showToast={showToast} />

      <LiveTelemetry data={data} showScores={!isCalibrating} />
    </>
  );
}

function ExamPanel({ data, showToast, setExamProgress, onExamFinished }) {
  const [questions, setQuestions] = useState([]);
  const [questionIdx, setQuestionIdx] = useState(0);
  const [answering, setAnswering] = useState(false);
  const [timeLeft, setTimeLeft] = useState(QUESTION_SECONDS);
  const [result, setResult] = useState(null);
  const [examReady, setExamReady] = useState(false);

  const ensureExamStarted = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/start_exam`);
      setQuestions(res.data.questions || []);
      setQuestionIdx(res.data.current_question_index || 0);
      setExamReady(true);
      return res.data;
    } catch (err) {
      setExamReady(false);
      showToast(err?.response?.data?.detail || "Exam requires completed calibration.", "error");
      return null;
    }
  }, [showToast]);

  useEffect(() => {
    setAnswering(false);
    setTimeLeft(QUESTION_SECONDS);
    setResult(null);
    ensureExamStarted();
  }, [ensureExamStarted]);

  useEffect(() => {
    if (answering) {
      setExamProgress((timeLeft / QUESTION_SECONDS) * 100);
      return;
    }
    setExamProgress(result ? 0 : 100);
  }, [answering, timeLeft, result, setExamProgress]);

  useEffect(() => () => setExamProgress(100), [setExamProgress]);

  useEffect(() => {
    if (!answering) return;
    const timer = setInterval(() => {
      setTimeLeft((seconds) => Math.max(0, seconds - 1));
    }, 1000);
    return () => clearInterval(timer);
  }, [answering]);

  const finalizeQuestion = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/end_question`);
      setResult(res.data);
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not compute question result.", "error");
    } finally {
      setAnswering(false);
    }
  }, [showToast]);

  useEffect(() => {
    if (!answering || timeLeft !== 0) return;
    finalizeQuestion();
  }, [answering, timeLeft, finalizeQuestion]);

  const startAnswer = async () => {
    setResult(null);
    setTimeLeft(QUESTION_SECONDS);
    try {
      const ready = await ensureExamStarted();
      if (!ready) return;
      await axios.get(`${API}/begin_answer`);
      setAnswering(true);
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not start answer timer.", "error");
    }
  };

  const nextQuestion = async () => {
    try {
      const res = await axios.get(`${API}/next_question`);
      setQuestionIdx(res.data.current_question_index);
      setResult(null);
      setTimeLeft(QUESTION_SECONDS);
      setAnswering(false);
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not load the next question.", "error");
    }
  };

  const redoQuestion = async () => {
    try {
      await axios.get(`${API}/redo_question`);
      setResult(null);
      setTimeLeft(QUESTION_SECONDS);
      setAnswering(false);
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not reset this question.", "error");
    }
  };

  const restartSession = async () => {
    try {
      await axios.get(`${API}/reset_exam`);
      setQuestions([]);
      setQuestionIdx(0);
      setAnswering(false);
      setTimeLeft(QUESTION_SECONDS);
      setResult(null);
      setExamReady(false);
      await ensureExamStarted();
      showToast("Interview session restarted.", "info");
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not restart the interview.", "error");
    }
  };

  const finishExam = async () => {
    try {
      const res = await axios.get(`${API}/end_exam`);
      onExamFinished(res.data);
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not end exam.", "error");
    }
  };

  const q = questions[questionIdx];
  const timer = useMemo(() => {
    const mm = String(Math.floor(timeLeft / 60)).padStart(2, "0");
    const ss = String(timeLeft % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  }, [timeLeft]);

  return (
    <>
      <div className="card">
        <h3>Question {questionIdx + 1} / 3</h3>
        <p>{q?.question || "Preparing interview questions..."}</p>
      </div>

      <div className="card">
        <h3>Timer</h3>
        <h1 className={`score-number ${timeLeft <= 20 && answering ? "critical" : "stable"}`}>{timer}</h1>
        <div className="exam-controls">
          <button className="exam-btn primary" onClick={startAnswer} disabled={!examReady || answering || Boolean(result)}>
            Start Answer
          </button>
          <button className="exam-btn secondary" onClick={finalizeQuestion} disabled={!answering}>
            End Now
          </button>
        </div>
      </div>

      <div className="card">
        <h3>Live Suggestion</h3>
        <div className="advice-box">{data?.suggestion || "Waiting..."}</div>
        {data?.is_cheating && answering ? <p style={{ color: "#f85149" }}>Focus warning: keep looking at the screen.</p> : null}
      </div>

      <AlgorithmPanel data={data} showToast={showToast} />

      {result ? (
        <div className="card">
          <h3>Question Score</h3>
          <h1 className={`score-number ${result.score < 70 ? "critical" : "stable"}`}>{result.score}/100</h1>
          <p>{result.label}</p>
          <p style={{ fontSize: "0.95em" }}>
            Posture quality: {result.posture_quality}% | Answer duration score: {result.duration_score}% | Time used: {result.elapsed_time}s
          </p>
          <div style={{ display: "grid", gap: 6 }}>
            {(result.errors || []).map((entry) => (
              <div key={entry.key} className="state-box">
                {entry.description} ({entry.percent_frames}%)
              </div>
            ))}
          </div>
          <div className="exam-controls result-actions" style={{ marginTop: 10 }}>
            <button className="exam-btn secondary" onClick={redoQuestion}>
              Redo Question
            </button>
            {questionIdx < 2 ? (
              <button className="exam-btn primary" onClick={nextQuestion}>Next Question</button>
            ) : (
              <>
                <button className="exam-btn secondary" onClick={restartSession}>
                  Restart Session
                </button>
                <button className="exam-btn primary" onClick={finishExam}>Finish Exam</button>
              </>
            )}
          </div>
        </div>
      ) : null}

      <LiveTelemetry data={data} showScores />
    </>
  );
}

function AlgorithmPanel({ data, showToast }) {
  const current = data?.algorithm?.toUpperCase?.() || "N/A";
  const available = Array.isArray(data?.available_algorithms) ? data.available_algorithms : [];

  const switchAlgorithm = async (algorithm) => {
    try {
      await axios.get(`${API}/set_algorithm`, { params: { algorithm } });
      showToast(`Switched RL policy to ${algorithm.toUpperCase()}.`, "success");
    } catch (err) {
      showToast(err?.response?.data?.detail || "Could not switch RL algorithm.", "error");
    }
  };

  return (
    <div className="card">
      <h3>RL Policy</h3>
      <p>Active model: <strong>{current}</strong></p>
      {data?.agent_runtime_error ? (
        <p style={{ color: "#f85149" }}>{data.agent_runtime_error}</p>
      ) : null}
      <div className="exam-controls">
        <button
          className="exam-btn secondary"
          onClick={() => switchAlgorithm("ppo")}
          disabled={!available.includes("ppo")}
        >
          Use PPO
        </button>
        <button
          className="exam-btn secondary"
          onClick={() => switchAlgorithm("dqn")}
          disabled={!available.includes("dqn")}
        >
          Use DQN
        </button>
      </div>
    </div>
  );
}

function SummaryPanel({ summary, onReturnHome }) {
  if (!summary) {
    return (
      <div className="summary-shell">
        <div className="card summary-card">
          <h3>Final Score</h3>
          <p>Summary not available.</p>
          <button className="exam-btn primary" onClick={onReturnHome}>Return Home</button>
        </div>
      </div>
    );
  }

  return (
    <div className="summary-shell">
      <div className="card summary-card">
        <h3>Final Score</h3>
        <h1 className={`score-number ${summary.score < 70 ? "critical" : "stable"}`}>{summary.score}/100</h1>
        <p>{summary.label}</p>
        <p>Average posture quality: {summary.posture_average}%</p>
        <p>Average answer duration score: {summary.duration_average}%</p>
        <p>Total speaking time: {summary.elapsed_time}s across {summary.questions_answered} questions</p>
        <div style={{ display: "grid", gap: 10, marginTop: 12 }}>
          {(summary.question_breakdown || []).map((entry) => (
            <div key={entry.question_index} className="state-box">
              Q{entry.question_index + 1}: {entry.score}/100, posture {entry.posture_quality}%, duration {entry.duration_score}%
            </div>
          ))}
        </div>
        <div className="exam-controls" style={{ marginTop: 16 }}>
          <button className="exam-btn primary" onClick={onReturnHome}>Return Home</button>
        </div>
      </div>
    </div>
  );
}

function LiveTelemetry({ data, showScores }) {
  const isCalibrating = data?.mode === "calibrating" || data?.mode === "calibration_freeze";
  return (
    <div className="card">
      <h3>Live Telemetry</h3>
      <p>State: {Array.isArray(data?.state) ? data.state.join(" | ") : "N/A"}</p>
      <p>Action: {data?.action || "N/A"}</p>
      <p>Reward: {showScores && !isCalibrating ? (data?.reward ?? 0) : "N/A"}</p>
      <p>Trust score: {showScores && !isCalibrating ? `${data?.trust_score ?? 0}%` : "N/A"}</p>
      <p>Looking away: {data?.is_cheating ? "Yes" : "No"}</p>
      <p>Source: {data?.identified_by || "N/A"}</p>
    </div>
  );
}

export default App;
