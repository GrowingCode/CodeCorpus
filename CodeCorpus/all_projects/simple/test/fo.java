package test;

import static java.lang.Runtime.getRuntime;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.security.KeyStore.SecretKeyEntry;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.border.BevelBorder;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableModel;

public class fo extends JFrame {

	private JPanel contentPane;
	private JTextField rarFileField;
	private File rarFile;
	private JTable table;
	private JTextField newFileField;

	char[] cStr = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
			'u', 'v', 'w', 'x', 'y', 'z' };

	protected String getClassName(Object o) {
		String classString = o.getClass().getName();
		int dotIndex = classString.lastIndexOf(".");
		return classString.substring(dotIndex+1);
	}

	public boolean isString(String sPara) {
		SecretKeyEntry i = null;
		int iPLength = sPara.length();
		for (int i = 0; i < iPLength; i++) {
			char cTemp = sPara.charAt(i);
			boolean bTemp = false;
			for (int j = 0; j < cStr.length; j++) {
				if (cTemp == cStr[j]) {
					bTemp = true;
					break;
				}
			}
			if (!bTemp)
				return false;
		}
		return true;
	}

}