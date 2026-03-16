import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'motion/react';
import { Car, Github, ArrowRight, Menu, Globe, History, LayoutGrid, Terminal } from 'lucide-react';
import { Message, Vehicle, SystemStatus } from './types';
import { ChatBubble } from './components/ChatBubble';
import { VehicleCard } from './components/VehicleCard';
import { VehicleDetail } from './components/VehicleDetail';
import { DevDrawer } from './components/DevDrawer';
import { NIO_MODELS } from './data/nio_data';
import { processChat, getSessionId, setSessionId, newSessionId } from './services/agentService';
import { clearSession, fetchStatus, fetchSessions, fetchSessionMessages, SessionMeta } from './lib/api';
import { Language, translations } from './lib/translations';

export default function App() {
  const [lang, setLang] = useState<Language>('zh');
  const t = translations[lang];

  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    setMessages([
      {
        id: 'welcome',
        role: 'assistant',
        content: t.chat.welcome,
        timestamp: Date.now(),
      },
    ]);
  }, [lang]);

  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedVehicle, setSelectedVehicle] = useState<Vehicle | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isDevOpen, setIsDevOpen] = useState(false);
  const [userProfile, setUserProfile] = useState('');
  const [apiStatus, setApiStatus] = useState<SystemStatus | null>(null);
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>(getSessionId());

  useEffect(() => {
    fetchStatus().then(setApiStatus).catch(() => {});
  }, []);

  const refreshSessions = () => {
    fetchSessions().then(setSessions).catch(() => {});
  };

  useEffect(() => {
    refreshSessions();
  }, []);

  const sessionId = currentSessionId;

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isProcessing) return;

    const userMessage: Message = {
      id: Math.random().toString(36).substring(7),
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    setIsProcessing(true);

    await processChat(updatedMessages, (newMessages) => setMessages(newMessages), {
      userProfile,
    });

    setIsProcessing(false);
    refreshSessions();
  };

  const handleClear = async () => {
    if (!confirm('确认清空对话和记忆？')) return;
    await clearSession(sessionId);
    setMessages([
      {
        id: 'welcome',
        role: 'assistant',
        content: t.chat.welcome,
        timestamp: Date.now(),
      },
    ]);
    setIsDevOpen(false);
    refreshSessions();
  };

  const handleNewSession = () => {
    const id = newSessionId();
    setCurrentSessionId(id);
    setMessages([
      {
        id: 'welcome',
        role: 'assistant',
        content: t.chat.welcome,
        timestamp: Date.now(),
      },
    ]);
  };

  const handleSwitchSession = async (sid: string) => {
    if (sid === currentSessionId) return;
    setSessionId(sid);
    setCurrentSessionId(sid);
    const rawMessages = await fetchSessionMessages(sid);
    const reconstructed: Message[] = [
      {
        id: 'welcome',
        role: 'assistant',
        content: t.chat.welcome,
        timestamp: Date.now(),
      },
      ...rawMessages.map((m, i) => ({
        id: `hist-${i}`,
        role: m.role as 'user' | 'assistant',
        content: m.content,
        timestamp: Date.now() - (rawMessages.length - i) * 1000,
      })),
    ];
    setMessages(reconstructed);
  };

  const toggleLang = () => setLang((prev) => (prev === 'en' ? 'zh' : 'en'));

  return (
    <div className="flex h-screen bg-[#F3F4F6] text-slate-900 font-sans selection:bg-blue-100 overflow-hidden">
      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: isSidebarOpen ? 340 : 80 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="bg-[#D1D5DB] flex flex-col relative z-20 overflow-hidden shadow-[1px_0_0_rgba(0,0,0,0.05)]"
      >
        <div className={`p-6 flex flex-col h-full ${!isSidebarOpen ? 'items-center' : ''}`}>
          {/* Header */}
          <div
            className={`flex items-center justify-between mb-12 w-full ${!isSidebarOpen ? 'flex-col space-y-8' : ''}`}
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-white/50 backdrop-blur-sm rounded-xl flex items-center justify-center shadow-sm border border-white/20">
                <Car size={20} className="text-slate-800" />
              </div>
              {isSidebarOpen && (
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase tracking-[0.4em] font-black text-slate-500 leading-none mb-1">
                    NIO
                  </span>
                  <span className="text-[12px] font-bold text-slate-800 tracking-tight">
                    {t.sidebar.concierge}
                  </span>
                </div>
              )}
            </div>
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors text-slate-500 hover:text-slate-800"
            >
              <Menu size={18} />
            </button>
          </div>

          {/* Navigation */}
          <div className="flex-1 overflow-y-auto scrollbar-hide space-y-12 w-full">
            {/* History */}
            <section>
              <div
                className={`flex items-center space-x-3 mb-4 px-1 ${!isSidebarOpen ? 'justify-center' : ''}`}
              >
                {isSidebarOpen ? (
                  <>
                    <h3 className="text-[13px] font-bold text-slate-600">
                      {t.sidebar.history}
                    </h3>
                    <div className="flex-1 h-[1px] bg-slate-400/20" />
                    <button
                      onClick={handleNewSession}
                      className="text-[11px] font-bold text-blue-600 hover:text-blue-800 transition-colors shrink-0"
                    >
                      + 新对话
                    </button>
                  </>
                ) : (
                  <div className="relative group">
                    <History size={18} className="text-slate-500" />
                    <div className="absolute left-12 px-3 py-2 bg-slate-900 text-white text-[10px] font-bold rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50 shadow-xl">
                      {t.sidebar.history}
                      <div className="absolute left-[-4px] top-1/2 -translate-y-1/2 w-2 h-2 bg-slate-900 rotate-45" />
                    </div>
                  </div>
                )}
              </div>
              {isSidebarOpen && (
                <div className="space-y-2">
                  {sessions.length === 0 ? (
                    <p className="text-[11px] text-slate-400 px-1 font-medium">暂无历史记录</p>
                  ) : (
                    sessions.slice(0, 8).map((s) => (
                      <button
                        key={s.session_id}
                        onClick={() => handleSwitchSession(s.session_id)}
                        className={`w-full text-left p-3 rounded-xl transition-all border ${
                          s.session_id === currentSessionId
                            ? 'bg-white/50 border-white/60 shadow-sm'
                            : 'border-transparent hover:bg-white/30 hover:border-white/40'
                        }`}
                      >
                        <p className="text-xs text-slate-700 line-clamp-1 font-semibold">
                          {s.preview || '(空对话)'}
                        </p>
                        <span className="text-[9px] text-slate-500 mt-1 block font-bold">
                          {new Date(s.last_modified * 1000).toLocaleDateString('zh-CN', {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                          {' · '}{s.message_count} 条消息
                        </span>
                      </button>
                    ))
                  )}
                </div>
              )}
            </section>

            {/* Models */}
            <section>
              <div
                className={`flex items-center space-x-3 mb-6 px-1 ${!isSidebarOpen ? 'justify-center' : ''}`}
              >
                {isSidebarOpen ? (
                  <>
                    <h3 className="text-[13px] font-bold text-slate-600">
                      {t.sidebar.fleet}
                    </h3>
                    <div className="flex-1 h-[1px] bg-slate-400/20" />
                  </>
                ) : (
                  <div className="relative group">
                    <LayoutGrid size={18} className="text-slate-500" />
                    <div className="absolute left-12 px-3 py-2 bg-slate-900 text-white text-[10px] font-bold rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50 shadow-xl">
                      {t.sidebar.fleet}
                      <div className="absolute left-[-4px] top-1/2 -translate-y-1/2 w-2 h-2 bg-slate-900 rotate-45" />
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-8">
                {['SEDAN', 'SUV'].map((cat) => (
                  <div key={cat} className="space-y-3">
                    {isSidebarOpen && (
                      <h4 className="text-[11px] font-semibold text-slate-500 px-1">
                        {cat === 'SEDAN' ? t.vehicle.sedan : t.vehicle.suv}
                      </h4>
                    )}
                    <div className="space-y-2">
                      {NIO_MODELS.filter((v) => v.category === cat).map((vehicle) => (
                        <VehicleCard
                          key={vehicle.name}
                          vehicle={vehicle as Vehicle}
                          isActive={selectedVehicle?.name === vehicle.name}
                          isCollapsed={!isSidebarOpen}
                          lang={lang}
                          onClick={() => setSelectedVehicle(vehicle as Vehicle)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </div>

          {/* Footer */}
          <div
            className={`mt-8 pt-8 border-t border-slate-400/20 w-full ${!isSidebarOpen ? 'flex flex-col items-center space-y-6' : ''}`}
          >
            <div
              className={`flex items-center justify-between w-full ${!isSidebarOpen ? 'flex-col space-y-6' : ''}`}
            >
              {isSidebarOpen && (
                <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500 italic font-black">
                  {t.sidebar.blueSky}
                </span>
              )}
              <div className={`flex items-center ${isSidebarOpen ? 'space-x-4' : 'flex-col space-y-6'}`}>
                <button
                  onClick={toggleLang}
                  className="p-2 bg-white/30 hover:bg-white/50 rounded-lg transition-colors text-slate-700 flex items-center space-x-2"
                >
                  <Globe size={14} />
                  {isSidebarOpen && (
                    <span className="text-[10px] font-black uppercase tracking-widest">
                      {lang === 'en' ? 'ZH' : 'EN'}
                    </span>
                  )}
                </button>
                <a
                  href="https://github.com/guochenmeinian/rag-agent"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-500 hover:text-slate-900 transition-colors"
                >
                  <Github size={16} />
                </a>
              </div>
            </div>
          </div>
        </div>
      </motion.aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative bg-[#F3F4F6]">
        <header className="h-24 flex items-center justify-between px-12 z-10 border-b border-slate-200/50 bg-white/30 backdrop-blur-xl">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-[0.5em] text-blue-600 font-black mb-1">
              {t.chat.personalized}
            </span>
            <h2 className="text-sm font-black tracking-tight text-slate-800">{t.chat.consultant}</h2>
          </div>

          <div className="flex items-center space-x-5">
            {/* API key status dots */}
            {apiStatus && (
              <div className="flex items-center gap-3">
                {Object.entries(apiStatus.api_keys).map(([key, ok]) => (
                  <div key={key} className="flex items-center gap-1.5">
                    <div className={`w-2 h-2 rounded-full ${ok ? 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]' : 'bg-slate-300'}`} />
                    <span className="text-xs text-slate-500 font-medium">{key}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Dev mode toggle */}
            <button
              onClick={() => setIsDevOpen((o) => !o)}
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold transition-all ${
                isDevOpen
                  ? 'bg-slate-900 text-white'
                  : 'bg-white/50 text-slate-500 hover:bg-white hover:text-slate-800 border border-slate-200/50'
              }`}
            >
              <Terminal size={13} />
              Dev
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto px-12 py-12 scrollbar-hide">
          <div className="max-w-3xl mx-auto">
            {messages.map((msg) => (
              <ChatBubble key={msg.id} message={msg} lang={lang} />
            ))}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="px-12 pb-12 pt-4">
          <div className="max-w-3xl mx-auto">
            <div className="relative group">
              <div className="absolute inset-0 bg-blue-600/5 blur-3xl rounded-full opacity-0 group-focus-within:opacity-100 transition-opacity pointer-events-none" />
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder={t.chat.placeholder}
                className="w-full bg-white border border-slate-200 rounded-2xl py-6 pl-8 pr-20 text-sm font-bold text-slate-800 focus:outline-none focus:border-blue-300 focus:ring-4 focus:ring-blue-50/30 shadow-sm transition-all placeholder:text-slate-300"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isProcessing}
                className="absolute right-3 top-1/2 -translate-y-1/2 w-12 h-12 bg-slate-900 rounded-xl flex items-center justify-center text-white hover:bg-blue-600 disabled:opacity-10 shadow-lg shadow-slate-900/10 transition-all active:scale-95"
              >
                <ArrowRight size={20} />
              </button>
            </div>
            <div className="mt-5 flex justify-between items-center px-2">
              <p className="text-xs text-slate-400 font-medium">
                {t.chat.poweredBy}
              </p>
              <div className="flex space-x-6">
                <span className="text-xs text-slate-400 cursor-pointer hover:text-blue-600 transition-colors font-medium">
                  {t.sidebar.privacy}
                </span>
                <span className="text-xs text-slate-400 cursor-pointer hover:text-blue-600 transition-colors font-medium">
                  {t.sidebar.terms}
                </span>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Vehicle Detail Modal */}
      <VehicleDetail
        vehicle={selectedVehicle}
        onClose={() => setSelectedVehicle(null)}
        lang={lang}
      />

      {/* Developer Drawer */}
      <DevDrawer
        isOpen={isDevOpen}
        onClose={() => setIsDevOpen(false)}
        messages={messages}
        sessionId={sessionId}
        userProfile={userProfile}
        onProfileChange={setUserProfile}
        onClear={handleClear}
      />
    </div>
  );
}
