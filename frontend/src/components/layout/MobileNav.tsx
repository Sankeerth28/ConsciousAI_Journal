import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  PenSquare,
  BookOpen,
  LineChart,
  Settings as SettingsIcon,
} from 'lucide-react';

export const MobileNav: React.FC = () => {
  const navItems = [
    { label: 'Dashboard', to: '/dashboard', icon: LayoutDashboard },
    { label: 'Write', to: '/journal/new', icon: PenSquare },
    { label: 'History', to: '/journal/history', icon: BookOpen },
    { label: 'Insights', to: '/insights', icon: LineChart },
    { label: 'Settings', to: '/settings', icon: SettingsIcon },
  ];

  return (
    <nav
      aria-label="Mobile Navigation"
      className="md:hidden fixed bottom-0 left-0 right-0 z-40 flex items-center justify-around h-16 border-t border-slate-800 bg-[#0c111e]/95 backdrop-blur-2xl px-2 shadow-2xl"
    >
      {navItems.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          className={({ isActive }) =>
            `flex flex-col items-center justify-center w-full py-1 gap-1 text-[11px] font-medium transition-colors ${
              isActive
                ? 'text-cyan-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`
          }
        >
          <item.icon className="h-5 w-5 shrink-0" />
          <span>{item.label}</span>
        </NavLink>
      ))}
    </nav>
  );
};
