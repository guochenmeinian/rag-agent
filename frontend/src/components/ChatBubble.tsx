import React from 'react';
import { motion } from 'motion/react';
import ReactMarkdown from 'react-markdown';
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

        <div className={`p-5 rounded-2xl shadow-sm border ${
          isUser
            ? 'bg-slate-900 text-white border-slate-800'
            : 'bg-white text-slate-700 border-slate-100'
        }`}>
          {message.content ? (
            isUser ? (
              <p className="text-[15px] leading-relaxed font-medium whitespace-pre-wrap text-left">{message.content}</p>
            ) : (
              <div className="prose prose-sm prose-slate max-w-none
                prose-headings:font-black prose-headings:text-slate-800 prose-headings:tracking-tight
                prose-h1:text-xl prose-h1:mb-3 prose-h1:mt-4
                prose-h2:text-lg prose-h2:mb-2 prose-h2:mt-5
                prose-h3:text-base prose-h3:mb-2 prose-h3:mt-4
                prose-p:text-[15px] prose-p:leading-relaxed prose-p:text-slate-700 prose-p:my-1.5
                prose-strong:text-slate-800 prose-strong:font-bold
                prose-ul:my-2 prose-ul:pl-0 prose-ul:space-y-1
                prose-ol:my-2 prose-ol:pl-4 prose-ol:space-y-1
                prose-li:text-[15px] prose-li:text-slate-700 prose-li:leading-relaxed prose-li:marker:text-slate-400
                prose-code:text-blue-600 prose-code:bg-blue-50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:font-mono prose-code:before:content-none prose-code:after:content-none
                prose-pre:bg-slate-900 prose-pre:text-slate-100 prose-pre:rounded-xl prose-pre:p-4 prose-pre:text-sm
                prose-hr:border-slate-100 prose-hr:my-4
                first:prose-p:mt-0 last:prose-p:mb-0">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>
            )
          ) : (
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
