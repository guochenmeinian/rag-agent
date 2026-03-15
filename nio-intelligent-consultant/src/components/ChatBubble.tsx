import React from 'react';
import { motion } from 'motion/react';
import { Message } from '../types';
import { Language, translations } from '../lib/translations';

interface ChatBubbleProps {
  message: Message;
  lang?: Language;
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({ message, lang = 'zh' }) => {
  const isUser = message.role === 'user';
  const t = translations[lang];

  const toolDisplayName = (name: string): string => {
    const key = name as keyof typeof t.tools;
    return t.tools[key] ?? name.replace(/_/g, ' ');
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex w-full mb-10 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`max-w-[85%] ${isUser ? 'text-right' : 'text-left'}`}>
        {!isUser && (
          <div className="flex items-center space-x-2 mb-3 px-1">
            <div className="w-1 h-1 bg-blue-600 rounded-full" />
            <span className="text-[10px] uppercase tracking-[0.3em] text-slate-400 font-extrabold">
              {t.chat.nioLabel}
            </span>
          </div>
        )}

        <div className={`p-5 rounded-2xl text-[15px] leading-relaxed shadow-sm border ${
          isUser
            ? 'bg-slate-900 text-white border-slate-800 font-medium'
            : 'bg-white text-slate-700 border-slate-100 font-normal'
        }`}>
          {message.content || (
            <div className="flex space-x-1.5 py-2">
              <div className="w-1.5 h-1.5 bg-slate-200 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-1.5 h-1.5 bg-slate-200 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-1.5 h-1.5 bg-slate-200 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          )}
        </div>

        {message.toolCalls && message.toolCalls.some(tc => tc.status === 'success') && !isUser && (
          <div className="mt-4 flex flex-wrap gap-2 px-1">
            {message.toolCalls.map((tool, idx) => (
              <div key={idx} className="text-[10px] tracking-wide text-blue-600 font-semibold border border-blue-100 bg-blue-50/50 px-3 py-1 rounded-lg">
                {toolDisplayName(tool.name)}
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
};
